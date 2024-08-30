import sys
sys.path.insert(0, '')

### Imports
import torch.nn as nn
import torch
from math import ceil
import time
from thop import profile
from fvcore.nn.flop_count import flop_count
from tqdm import tqdm

from models.ms_tcn_1D import MultiScale_TemporalConv
from models.trajectory_self_attention import Trajectory_SelfAttention # DETR-like self-attention
from utils import count_params

# Notations
# N: batch size
# C: 2 or 3 (channel dimension/#coordinates)
# T: #frames
# J: #joints
# E: #entities (2 if hands are separated)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class MicroactionEncoder(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=256, num_heads=[16,32,64], dropout=0.2, num_frames=15, \
                    num_joints=21, num_hands=2, stride=[1,2,2], kernel_size=3, dilations=[1,2], \
                    global_wrist_ref=False):
        super().__init__()

        """
        This model encodes a microaction (pose) with 3 temporal convolution layers and 3 trajectory self-attention layers interleaved.
        :param input_dim: int, input dimension of the pose data (3 for 3D coordinates).
        :param embedding_dim: int, output feature dimension for the microaction.
        :param num_heads: list of 3 ints, number of heads for each trajectory self-attention layer.
        :param dropout: float, dropout rate.
        :param num_frames: int, number of frames in the microaction. Defined by the microaction length / window size.
        :param num_joints: int, number of joints in the pose data.
        :param num_hands: int, number of hands in the pose data.
        :param stride: list of 3 ints, stride for each temporal convolution layer.
        :param kernel_size: int, kernel size for the temporal convolution layers.
        :param dilations: list of 2 ints, dilation for the scales of multi-scale TCNs. same for all layers.
        :param global_wrist_ref: bool, whether to calculate and use the Global Wrist Token for the microaction encoding.
        """

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Channel dimensions grow, e.g., 3->64->128->256
        self.C1 = embedding_dim // 4
        self.C2 = embedding_dim // 2
        self.C3 = embedding_dim

        # Temporal dimensions shrink, e.g., 15->8->4
        self.T1 = int(ceil(num_frames/stride[0]))
        self.T2 = int(ceil(self.T1/stride[1]))
        self.T3 = int(ceil(self.T2/stride[2]))

        num_nodes = num_joints * num_hands + (1 if global_wrist_ref else 0) # Number of tokens for self-attention with/without Global Wrist Token.

        self.data_bn = nn.BatchNorm1d(num_hands*num_joints*input_dim) # MVC, e.g., 2*21*3

        # First TCN+SA layer
        self.tcn1 = nn.Sequential(
                                MultiScale_TemporalConv(input_dim, self.C1,
                                    kernel_size=kernel_size,
                                    stride=stride[0],
                                    dilations=dilations),
                                MultiScale_TemporalConv(self.C1, self.C1,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    dilations=dilations),
                                MultiScale_TemporalConv(self.C1, self.C1,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    dilations=dilations),
        )
        # self.bn1 = nn.BatchNorm3d(self.C1)
        self.att1 = Trajectory_SelfAttention(self.C1, self.T1, num_nodes, num_heads[0], dropout_rate=dropout)

        # Second TCN+SA layer
        self.tcn2 = nn.Sequential(
                                MultiScale_TemporalConv(self.C1, self.C2,
                                    kernel_size=kernel_size,
                                    stride=stride[1],
                                    dilations=dilations),
                                MultiScale_TemporalConv(self.C2, self.C2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    dilations=dilations),
                                MultiScale_TemporalConv(self.C2, self.C2,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    dilations=dilations)
        )

        # self.bn2 = nn.BatchNorm3d(self.C2)
        self.att2 = Trajectory_SelfAttention(self.C2, self.T2, num_nodes, num_heads[1], dropout_rate=dropout)

        # Third TCN+SA layer
        self.tcn3 = nn.Sequential(
                                MultiScale_TemporalConv(self.C2, self.C3,
                                    kernel_size=kernel_size,
                                    stride=stride[2],
                                    dilations=dilations),
                                MultiScale_TemporalConv(self.C3, self.C3,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    dilations=dilations),
                                MultiScale_TemporalConv(self.C3, self.C3,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    dilations=dilations)
        )
        
        # self.bn3 = nn.BatchNorm3d(self.C3)
        self.att3 = Trajectory_SelfAttention(self.C3, self.T3, num_nodes, num_heads[2], dropout_rate=dropout)
        
        self.global_wrist_ref = global_wrist_ref
        if global_wrist_ref:
            # Global Wrist Token is calculated once (at embedding_dim) and projected to all layers
            self.wrist_proj1 = nn.Linear(embedding_dim,self.C1)
            self.wrist_proj2 = nn.Linear(embedding_dim,self.C2)
            self.wrist_proj3 = nn.Linear(embedding_dim,self.C3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x, global_wrist_token=None):
        N, C, T, J, E = x.shape

        # # [Optional] Normalize using the mean location of wrist to normalize the rest of the joints
        # x_wrist = x[:,:,:,5,:].mean(dim=2) # Take wrist (index=5), average over T --> (B, 3, 2) -- N C E
        # x = x - x_wrist.unsqueeze(2).unsqueeze(3) # (N, C, 1, 1, E)
        
        if self.global_wrist_ref:
            """
            If global_wrist_ref is True, global_wrist_token should be provided.
            global_wrist_token is calculated over the full action clip with wrist features only (location and/or orientation).
            """
            layer1_fullWrist_feat = self.wrist_proj1(global_wrist_token)
            layer2_fullWrist_feat = self.wrist_proj2(global_wrist_token)
            layer3_fullWrist_feat = self.wrist_proj3(global_wrist_token)

        x = self.data_bn(x.permute(0,4,3,1,2).contiguous().view(N,E*J*C,T)).view(N,E,J,C,T).permute(0,3,4,2,1).contiguous() # NCTJE

        # First TCN+SA layer
        x = self.tcn1(x.permute(0,3,4,1,2).contiguous().view(N*J*E,C,T)).view(N, J, E, self.C1, self.T1).permute(0,3,4,1,2).contiguous() # TCN applied and then back to NCTJE
        # x = self.bn1(x) # [N, 64, 15, 21, 2].
        if self.global_wrist_ref:        
            num_segments = x.shape[0] // layer1_fullWrist_feat.shape[0] # N = original batch size * number of microactions. Original batch size comes from global_wrist_token.
            layer1_fullWrist_feat = layer1_fullWrist_feat.unsqueeze(1).repeat(1,num_segments,1).view(-1,self.C1) # Repeat wrist feat to match batch size considering #microactions
            layer1_fullWrist_feat = layer1_fullWrist_feat.unsqueeze(-1).repeat(1,1,self.T1).unsqueeze(-1) # Adjust temporal dim. Final shape: (N(=B*8), 64, 15, 1)
            x = torch.cat((x.view(N, self.C1, self.T1, J*E), layer1_fullWrist_feat), dim=-1)
            x = self.att1(x)[:,:,:,:J*E].view(N, self.C1, self.T1, J, E) # Back to NCTJE
        else:
            x = self.att1(x.view(N, self.C1, self.T1, J*E)).view(N, self.C1, self.T1, J, E) # Back to NCTJE

        # Second TCN+SA layer
        x = self.tcn2(x.permute(0,3,4,1,2).contiguous().view(N*J*E,self.C1,self.T1)).view(N, J, E, self.C2, self.T2).permute(0,3,4,1,2).contiguous() # TCN applied and then back to NCTJE
        # x = self.bn2(x) # [N, 128, 8, 21, 2]
        if self.global_wrist_ref:
            num_segments = x.shape[0] // layer2_fullWrist_feat.shape[0]
            layer2_fullWrist_feat = layer2_fullWrist_feat.unsqueeze(1).repeat(1,num_segments,1).view(-1,self.C2)
            layer2_fullWrist_feat = layer2_fullWrist_feat.unsqueeze(-1).repeat(1,1,self.T2).unsqueeze(-1)
            x = torch.cat((x.view(N, self.C2, self.T2, J*E), layer2_fullWrist_feat), dim=-1)
            x = self.att2(x)[:,:,:,:J*E].view(N, self.C2, self.T2, J, E) # Back to NCTJE
        else:
            x = self.att2(x.view(N, self.C2, self.T2, J*E)).view(N, self.C2, self.T2, J, E) # Back to NCTJE

        # Third TCN+SA layer
        x = self.tcn3(x.permute(0,3,4,1,2).contiguous().view(N*J*E,self.C2,self.T2)).view(N, J, E, self.C3, self.T3).permute(0,3,4,1,2).contiguous() # TCN applied and then back to NCTJE
        # x = self.bn3(x) # [N, 256, 4, 21, 2]
        if self.global_wrist_ref:
            num_segments = x.shape[0] // layer3_fullWrist_feat.shape[0]
            layer3_fullWrist_feat = layer3_fullWrist_feat.unsqueeze(1).repeat(1,num_segments,1).view(-1,self.C3)
            layer3_fullWrist_feat = layer3_fullWrist_feat.unsqueeze(-1).repeat(1,1,self.T3).unsqueeze(-1)
            x = torch.cat((x.view(N, self.C3, self.T3, J*E), layer3_fullWrist_feat), dim=-1)
            x = self.att3(x)[:,:,:,:J*E].view(N, self.C3, self.T3, J, E) # Back to NCTJE
        else:
            x = self.att3(x.view(N, self.C3, self.T3, J*E)).view(N, self.C3, self.T3, J, E) # Back to NCTJE

        # Take global spatio-temporal average for 256-dim token
        out = x.mean(dim=[2,3,4])
        return out

if __name__ == "__main__":
    import sys
    sys.path.append('..')

    microaction_window_size = 15
    model = MicroactionEncoder(embedding_dim=256, num_heads=[16,32,64], dropout=0.2, num_frames=microaction_window_size,\
                               num_joints=21, num_hands=2, stride=[1,2,2], kernel_size=3, dilations=[1,2], global_wrist_ref=False).cuda()

    N, C, T, V, M = 1, 3, microaction_window_size, 21, 2
    x = torch.randn(N,C,T,V,M).cuda()
    out = model(x)

    print(out.shape)
    print('Model total # params:', count_params(model))

    ### Efficiency metrics

    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9} GFLOPs")
    print("#param: ", params)

    num_samples = 100  # Adjust as needed
    total_time = 0

    for _ in tqdm(range(num_samples)):
        start_time = time.time()
        with torch.no_grad():
            _ = model(x,) #, rgb_batch)
        end_time = time.time()
        total_time += end_time - start_time

    average_inference_time = total_time / num_samples
    print(f"Average Inference Time: {average_inference_time} seconds")

    gflops = flops / (average_inference_time * 1e9)
    print(f"GFLOPS: {gflops} GFLOPs/s")
    
    gflop_dict, _ = flop_count(model, (x,))
    gflops = sum(gflop_dict.values())
    print("GFLOPs: ", gflops)