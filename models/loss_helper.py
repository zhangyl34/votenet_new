import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from my_util import nn_distance, huber_loss

OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

def compute_objectness_loss(end_points):
    """ 
    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
    """ 

    # 获取 label 和 mask
    objectness_label = end_points['vote_label_mask'].contiguous()     # (B,N)
    objectness_mask = end_points['vote_label_mask_mask'].contiguous() # (B,N)

    # 计算 objectness loss
    objectness_scores = end_points['pred_seg']  # (B,N,2)
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    return objectness_loss

def compute_box_loss(end_points):
    """ Compute 3D bounding box loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
    """

    # 获取 kpts offset
    kp_targ_ofst = end_points['vote_label'].contiguous()  # (B,kp,N,3)
    bs = kp_targ_ofst.shape[0]
    kp_num = kp_targ_ofst.shape[1]
    n_pts = kp_targ_ofst.shape[2]
    objectness_label = (end_points['vote_label_mask'] > 1e-8).float() # B,N
    objectness_label = objectness_label.view(bs,1,n_pts,1).repeat(1,kp_num,1,1).contiguous()  # B,kp,N,1

    # 计算 kpts loss
    pred_ofsts = end_points['pred_kp_of']  # (B,kp,N,3) offset
    abs_diff = objectness_label * torch.abs(pred_ofsts - kp_targ_ofst)  # B,kp,N,3
    kpts_loss = torch.sum(abs_diff.view(bs,kp_num,-1),2) / (torch.sum(objectness_label.view(bs,kp_num,-1),2)+1e-3)
    kpts_loss = kpts_loss.sum()
    return kpts_loss


def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Obj loss
    objectness_loss = compute_objectness_loss(end_points)
    objectness_loss *= 30
    end_points['objectness_loss'] = objectness_loss

    kpts_loss = compute_box_loss(end_points)
    end_points['kpts_loss'] = kpts_loss
    
    loss = objectness_loss + kpts_loss
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    # obj_pred_val = torch.argmax(end_points['objectness_scores'], 2)  # B,K
    # obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float())/(obj_pred_val.shape[0]*obj_pred_val.shape[1])
    # end_points['obj_acc'] = obj_acc

    return loss, end_points
