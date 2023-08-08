""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from loss_helper import get_loss
from dump_helper import dump_results
from pytorch_utils import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    FC,
)

class Seq(nn.Sequential):
    def __init__(self, input_channels):
        super(Seq, self).__init__()
        self.count = 0
        self.current_channels = input_channels

    def conv1d(
        self,
        out_size,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm1d,
    ):
        # type: (Seq, int, int, int, int, int, Any, bool, Any, bool, bool, AnyStr) -> Seq

        self.add_module(
            str(self.count),
            Conv1d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def conv2d(
        self,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Seq, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr) -> Seq

        self.add_module(
            str(self.count),
            Conv2d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def conv3d(
        self,
        out_size,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm3d,
    ):
        # type: (Seq, int, Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Any, bool, Any, bool, bool, AnyStr) -> Seq

        self.add_module(
            str(self.count),
            Conv3d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def fc(
        self,
        out_size,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=None,
        preact=False,
        name="",
    ):
        # type: (Seq, int, Any, bool, Any, bool, AnyStr) -> None

        self.add_module(
            str(self.count),
            FC(
                self.current_channels,
                out_size,
                activation=activation,
                bn=bn,
                init=init,
                preact=preact,
                name=name,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def dropout(self, p=0.5):
        # type: (Seq, float) -> Seq

        self.add_module(str(self.count), nn.Dropout(p=0.5))
        self.count += 1

        return self

    def maxpool2d(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        self.add_module(
            str(self.count),
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            ),
        )
        self.count += 1

        return self

class Deeperception(nn.Module):
    def __init__(self, num_points):
        super(Deeperception(num_points), self).__init__()
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, end_points):
        cld_emb = end_points['fp4_features']
        bs, _, n_pts = cld_emb.size()
        feat_1 = cld_emb  # B,128,N
        feat_2 = F.relu(self.conv2_cld(cld_emb))  # B,256,N

        feat_3 = F.relu(self.conv3(feat_1))
        feat_3 = F.relu(self.conv4(feat_3))
        ap_x = self.ap1(feat_3)
        ap_x = ap_x.view(-1, 512, 1).repeat(1, 1, n_pts)  # B,512,N

        end_points['deeper_features'] = torch.cat([feat_1, feat_2, ap_x], 1)  # 128 + 256 + 512 = 896

        return end_points

class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, input_feature_dim=0, num_points=7000, sampling='vote_fps'):
        super().__init__()

        # self.num_heading_bin = num_heading_bin      # 12
        self.input_feature_dim = input_feature_dim  # 3
        self.num_points = num_points                # 7000
        self.sampling = sampling                    # random
        self.num_kps = 8

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # deeper perception
        self.deeper_feat = Deeperception(num_points)

        # segmentation
        self.SEG_layer = (
            Seq(896)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(2, activation=None)
        )

        self.KpOF_layer = (
            pt_utils.Seq(896)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(256, bn=True, activation=nn.ReLU())
            .conv1d(self.num_kps*3, activation=None)
        )

        # # Hough voting
        # self.vgen = VotingModule(256)

        # # Vote aggregation and detection
        # self.pnet = ProposalModule(num_heading_bin, num_proposal, sampling)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], inputs['pcd_normal'], end_points)

        end_points = self.deeper_feat(end_points)  # B,896,N

        # segmentation
        pred_seg = self.SEG_layer(end_points['deeper_features']).transpose(1, 2).contiguous()  # B,N,2
        end_points['pred_seg'] = pred_seg

        # key points prediction
        pred_kp_of = self.KpOF_layer(end_points['deeper_features']).view(batch_size, self.num_kps, 3, N)
        pred_kp_of = pred_kp_of.permute(0, 1, 3, 2).contiguous()  # B,8,N,3
        end_points['pred_kp_of'] = pred_kp_of

        return end_points


