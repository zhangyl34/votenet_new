""" Training routine for 3D object detection.
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

from autolab_core import Logger, RigidTransform
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_points', type=int, default=7000, help='downsample points number')
parser.add_argument('--cluster_sampling', default='random', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random')
parser.add_argument('--max_epoch', type=int, default=190, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='40,80,120', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
sys.path.append(os.path.join(ROOT_DIR, 'syndata'))
from synthetic_dataset import SyntheticDataset
from synthetic_dataset_new import SyntheticDatasetNew
from data_config import DatasetConfig
DATASET_CONFIG = DatasetConfig()
TRAIN_DATASET = SyntheticDataset('train', augment=True, num_points=FLAGS.num_points)
TEST_DATASET = SyntheticDataset('val', augment=False, num_points=FLAGS.num_points)
TEST_DATASET_NEW = SyntheticDatasetNew('demo', augment=False)

print("dataset size:", len(TRAIN_DATASET), len(TEST_DATASET))  # 数据集大小
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER_NEW = DataLoader(TEST_DATASET_NEW, batch_size=1,
    shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)
print("iterations per epoch:", len(TRAIN_DATALOADER), len(TEST_DATALOADER),  len(TEST_DATALOADER_NEW))  # data size / batch size = iteration

# Init the model and optimzier
MODEL = importlib.import_module('votenet') # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = 3

Detector = MODEL.VoteNet

net = Detector(input_feature_dim=num_input_channel,
               num_points=FLAGS.num_points,
               sampling=FLAGS.cluster_sampling)

if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(FLAGS, 'train')
TEST_VISUALIZER = TfVisualizer(FLAGS, 'test')

# Used for AP calculation
CONFIG_DICT = {'conf_thresh':0.0005, 'dataset_config':DATASET_CONFIG}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):  # iteration (4000/8=500)

        # 返回了 8 组样本数据
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
    
        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)

        # temp
        # print(batch_data_label['scan_name'])
        # pred_map_cls = parse_predictions(end_points, CONFIG_DICT)  # prob,x,y,z,euler
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            # 名字中包含 'loss','acc','ratio' 的 key
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
                (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

# evaluate old version
# def evaluate_one_epoch():
#     stat_dict = {}  # collect statistics
#     # net.train()     # set model to training mode
#     net.eval()      # set model to eval mode (for bn and dp)
#     for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
#         if batch_idx % 10 == 0:
#             print('Eval batch: %d'%(batch_idx))
#         for key in batch_data_label:
#             batch_data_label[key] = batch_data_label[key].to(device)
        
#         # Forward pass
#         inputs = {'point_clouds': batch_data_label['point_clouds']}
#         with torch.no_grad():
#             end_points = net(inputs)

#         # Compute loss
#         for key in batch_data_label:
#             assert(key not in end_points)
#             end_points[key] = batch_data_label[key]
#         loss, end_points = criterion(end_points, DATASET_CONFIG)

#         # Accumulate statistics and print out
#         for key in end_points:
#             if 'loss' in key or 'acc' in key or 'ratio' in key:
#                 if key not in stat_dict: stat_dict[key] = 0
#                 stat_dict[key] += end_points[key].item()

#         # print(batch_data_label['scan_name'])
#         # batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
#         # batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
#         # ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

#         # Dump evaluation results for visualization
#         # if FLAGS.dump_results and batch_idx == 0:
#         #     print("i'm dumping results...")
#         #     MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG) 

#     # Log statistics
#     # TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
#     #     (EPOCH_CNT+1)*len(TEST_DATALOADER)*BATCH_SIZE)
#     for key in sorted(stat_dict.keys()):
#         log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

#     # Evaluate average precision
#     # metrics_dict = ap_calculator.compute_metrics()
#     # for key in metrics_dict:
#     #     log_string('eval %s: %f'%(key, metrics_dict[key]))

#     mean_loss = stat_dict['loss']/float(batch_idx+1)
#     return mean_loss


# evaluate new version
def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    # net.train()     # set model to training mode
    net.eval()      # set model to eval mode (for bn and dp)

    gt_data = np.loadtxt('data/data_demo/end_pose_ref.log')

    Tm_4 = RigidTransform(
        rotation=RigidTransform.z_axis_rotation((45)*np.pi/180),
        translation=np.array([0.0,0.0,0.0]),
        from_frame="marker_4",
        to_frame="base",
    )
    Tm_5 = RigidTransform(
        rotation=RigidTransform.x_axis_rotation(0),
        translation=np.array([0, 0, -4.5]),  #-(- 4.237 + 6.3 + 1.3)
        from_frame="marker_5",
        to_frame="marker_4",
    )

    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER_NEW):
        
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        i = batch_data_label['scan_idx']
        T_qua2rota = RigidTransform(
            rotation=np.array([gt_data[i][3], gt_data[i][4], gt_data[i][5], gt_data[i][6]]),
            translation=np.array([gt_data[i][0], gt_data[i][1], gt_data[i][2]]),
            from_frame="marker_5",
            to_frame="world",
        )
        pose_gt = T_qua2rota*(Tm_4*Tm_5).inverse()

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        # batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        # ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # 计算误差
        print(pose_gt.translation)
        print(R.from_matrix(pose_gt.rotation).as_euler('XYZ'))
        position_error = np.sqrt(np.sum((pose_gt.translation-pred_map_cls[0][0][1][0:3])*(pose_gt.translation-pred_map_cls[0][0][1][0:3])))
        R_predict = R.from_euler('XYZ', pred_map_cls[0][0][1][3:6]).as_matrix()
        R_gt = pose_gt.rotation
        z_predict = R_predict[:,2]
        z_gt = R_gt[:,2]
        z_error = np.degrees(np.arccos(np.dot(z_predict, z_gt)))
        x_predict = R_predict[:,0]
        x_gt = R_gt[:,0]
        x_error = np.degrees(np.arccos(np.dot(x_predict, x_gt)))

        if len(pred_map_cls[0]) > 0:
            log_string('position errror: %f'%(position_error))
            log_string('z_axis error: %f'%(z_error))
            log_string('x_axis error: %f'%(x_error))

    return 0

def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    # evaluate_one_epoch()
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        # if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
        #     loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)
