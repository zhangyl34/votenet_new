""" Demo of using VoteNet 3D object detector to detect objects from a point cloud.
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR  # votenet
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from ap_helper import parse_predictions
from pc_util import random_sampling, read_ply, write_ply
from my_util import softmax
from synthetic_dataset import DC  # dataset config
import data_config
import open3d as o3d
from autolab_core import Logger, RigidTransform
from scipy.spatial.transform import Rotation as R
from scipy import optimize


LOG_FOUT = open(os.path.join(BASE_DIR, 'data/data_demo/results/log_train.txt'), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def preprocess_point_cloud(point_cloud, R_intuitive, t_intuitive):
    ''' Prepare the numpy point cloud (N,6) for forward pass '''
    point_cloud = point_cloud

    if (point_cloud.shape[1]==3):
        normal = get_normal(point_cloud, R_intuitive, t_intuitive)
        point_cloud = np.concatenate((point_cloud, normal), axis=1)

    pcd_num = 7000
    if point_cloud.shape[0] >= pcd_num:
        choice = np.random.choice(point_cloud.shape[0], pcd_num, replace=False)
        point_cloud = point_cloud[choice, :]
    else:
        choice = np.array([i for i in range(point_cloud.shape[0])])
        choice = np.pad(choice, (0, pcd_num-len(choice)), 'wrap')
        point_cloud = point_cloud[choice, :]

    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,num_points,6)
    return pc

def get_normal(cld, R_intuitive, t_intuitive):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cld)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    n = np.asarray(pcd.normals)

    # 每个点对 x 轴投票的真值
    new_cld = np.dot(R_intuitive.T, cld.T-t_intuitive)  # 3,N
    new_theta = np.arctan2(new_cld[1,:], new_cld[0,:])  # N,
    new_normal = np.zeros((new_theta.shape[0], 3))
    new_normal[np.where((new_theta<=np.pi/4)*(new_theta>-np.pi/4)),:] = np.array([1,0,0])
    new_normal[np.where((new_theta<=3*np.pi/4)*(new_theta>np.pi/4)),:] = np.array([0,1,0])
    new_normal[np.where((new_theta<=-np.pi/4)*(new_theta>-3*np.pi/4)),:] = -np.array([0,1,0])
    new_normal[np.where((new_theta<=-3*np.pi/4)+(new_theta>3*np.pi/4)),:] = -np.array([1,0,0])
    normal = np.dot(R_intuitive, new_normal.T)  # (3,N)

    if (np.sum(np.sum(n*normal.T, axis=1))<0):
        n = -n
    
    return n

def fun(x, rmin, theta_min):
    # 点的极坐标，theta_min: [0,pi/2]; theta_min-x: [-pi/2,pi/2]
    v = np.sum(((theta_min-x)>0)*((rmin*np.cos(theta_min-x)-np.sqrt(2)/2)**2+(rmin*np.sin(theta_min-x)-np.sqrt(2)/2)**2), axis=0) + \
        np.sum(((theta_min-x)<0)*((rmin*np.cos(theta_min-x)-np.sqrt(2)/2)**2+(rmin*np.sin(theta_min-x)+np.sqrt(2)/2)**2), axis=0)
    return v

if __name__=='__main__':

    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'data/data_demo') 
    checkpoint_path = os.path.join(demo_dir, 'checkpoint.tar')
    # ['006975', '007770', ...] 从低到高排序
    scan_names = sorted(list(set([os.path.basename(x)[0:4] \
        for x in os.listdir(os.path.join(demo_dir, 'ply'))])))
    gt_data = np.loadtxt(demo_dir + '/end_pose_ref.log')

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    net = MODEL.VoteNet(input_feature_dim=3,
               num_points=7000,
               sampling='random')
    net.to(device)
    print('Constructed model.')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))
    net.eval()  # eval model will lead to bad predict, don't know why

    Tm_4 = RigidTransform(
        rotation=RigidTransform.z_axis_rotation((-150)*np.pi/180),  # dataset1 +45
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

    error = np.zeros((len(scan_names),2))
    for i in range(len(scan_names)):

        scan_name = scan_names[i]

        T_qua2rota = RigidTransform(
            rotation=np.array([gt_data[int(scan_name)-1][3], gt_data[int(scan_name)-1][4], gt_data[int(scan_name)-1][5], gt_data[int(scan_name)-1][6]]),
            translation=np.array([gt_data[int(scan_name)-1][0], gt_data[int(scan_name)-1][1], gt_data[int(scan_name)-1][2]]),
            from_frame="marker_5",
            to_frame="world",
        )
        pose_gt = T_qua2rota*(Tm_4*Tm_5).inverse()

        # Load and preprocess input point cloud
        point_cloud = read_ply(demo_dir+'/ply/'+scan_name+'.ply')
        # point_cloud = np.load(demo_dir + '/ply/' +  scan_name + '_pc.npz')['pc']
        R_intuitive = R.from_matrix(pose_gt.rotation).as_matrix()
        t_intuitive = pose_gt.translation.reshape(3,1)
        pc = preprocess_point_cloud(point_cloud, R_intuitive, t_intuitive)
        assert(pc.shape[2]==6)
        print('Loaded point cloud data: %s'%(scan_name))
    
        # Model inference
        inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
        with torch.no_grad():
            end_points = net(inputs)
        
        # write key points and label points
        obj_threshold = 0.99
        pred_ofsts = end_points['pred_kp_of']  # 1,2,7k,3
        obj_prob = softmax(end_points['pred_seg'].detach().cpu().numpy())[0,:,1]  # 1,7k,2 -> 7k,
        pred_ofsts = pred_ofsts[:,:,:,:].cpu().numpy()
        pred_z = pred_ofsts[0,0,np.where(obj_prob>obj_threshold),:].reshape(-1,3)  # N,3
        pred_x = pred_ofsts[0,1,np.where(obj_prob>obj_threshold),:].reshape(-1,3)  # N,3

        dump_dir = os.path.join(demo_dir, 'results/' + scan_name)
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        write_ply(pred_z, os.path.join(dump_dir, 'z.ply'))
        write_ply(pred_x, os.path.join(dump_dir, 'x.ply'))
        write_ply(pc[0,np.where(obj_prob>obj_threshold),0:3].reshape(-1,3), os.path.join(dump_dir, 'raw.ply'))

        # 计算误差
        R_gt = pose_gt.rotation
        z_gt = R_gt[:,2]
        x_gt = R_gt[:,0]
        # 半径滤波
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_x)
        _, filtIndex = pcd.remove_radius_outlier(nb_points=100,radius=0.05)  # 100 or 10?
        filtpcd = pcd.select_by_index(filtIndex)
        # 拟合正方形
        points = np.array(filtpcd.points)
        norm = z_gt
        z_new = np.array([0,0,1])-norm[2]*norm
        z_new = z_new / np.linalg.norm(z_new)
        y_new = norm
        x_new = np.cross(y_new,z_new)
        x_new = x_new / np.linalg.norm(x_new)
        Projection = np.dot(np.array([[0,0,1],[1,0,0]]), np.concatenate([x_new.reshape(1,3),y_new.reshape(1,3),z_new.reshape(1,3)],axis=0))
        projpcd = np.dot(Projection,points.T).T
        rmin = np.sqrt(np.sum(projpcd*projpcd, axis=1))
        theta_min = np.arctan2(projpcd[:,1], projpcd[:,0])  # [-pi,pi]
        theta_min = theta_min - (theta_min>np.pi/2)*np.pi/2
        theta_min = theta_min + (theta_min<0)*np.pi/2
        theta_min = theta_min + (theta_min<0)*np.pi/2  # [0,pi/2]
        res = optimize.minimize_scalar(fun, args=(rmin,theta_min), bounds=(0,np.pi/2), method='bounded')
        # 计算误差
        u_predict1 = np.cos(res.x-np.pi/4)  # [-pi/4,pi/4]
        v_predict1 = np.sin(res.x-np.pi/4)
        x_predict1 = z_new*u_predict1 + x_new*v_predict1
        x_error1 = np.degrees(np.arccos(np.dot(x_predict1, x_gt)))
        u_predict2 = np.cos(res.x+np.pi/4)  # [pi/4,3*pi/4]
        v_predict2 = np.sin(res.x+np.pi/4)
        x_predict2 = z_new*u_predict2 + x_new*v_predict2
        x_error2 = np.degrees(np.arccos(np.dot(x_predict2, x_gt)))
        u_predict3 = np.cos(res.x-3*np.pi/4)  # [-3*pi/4,-pi/4]
        v_predict3 = np.sin(res.x-3*np.pi/4)
        x_predict3 = z_new*u_predict3 + x_new*v_predict3
        x_error3 = np.degrees(np.arccos(np.dot(x_predict3, x_gt)))
        u_predict4 = np.cos(res.x+3*np.pi/4)  # [-5*pi/4,-3*pi/4]
        v_predict4 = np.sin(res.x+3*np.pi/4)
        x_predict4 = z_new*u_predict4 + x_new*v_predict4
        x_error4 = np.degrees(np.arccos(np.dot(x_predict4, x_gt)))
        x_error = np.min([x_error1,x_error2,x_error3,x_error4])
        
        # 保存结果
        error[i][0] = 0
        error[i][1] = x_error
        log_string('z_axis error: %f'%(0))
        log_string('x_axis error: %f'%(x_error))
    np.save('error.npy', error)




