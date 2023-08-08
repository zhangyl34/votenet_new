''' Voting module: generate votes from XYZ and features of seed points.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class VotingModule(nn.Module):
    def __init__(self, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.in_dim = seed_feature_dim  # 256
        self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim), 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)  #, track_running_stats=False)  # for little batch
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)  #, track_running_stats=False)  # for little batch
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net)  # (batch_size, (3+out_dim), num_seed)

        net = net.transpose(2,1).view(batch_size, num_seed, 1, 3+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        
        residual_features = net[:,:,:,3:]  # (batch_size, num_seed, 1, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features
 

