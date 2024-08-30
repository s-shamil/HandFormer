### For pose+rgb experiments ###

import sys
sys.path.insert(0, '')

import torch.nn as nn
import torch
import torch.nn.functional as F
from thop import profile
from fvcore.nn.flop_count import flop_count
from tqdm import tqdm
import time

from utils import count_params
from models.microaction_encoder import MicroactionEncoder
from models.ms_tcn_1D import MultiScale_TemporalConv as MS_TCN
from models.transformer_bimodal import Bimodal_TF
from models.hf_pose import HF_Pose
from models.mlp_1D import MLP

# Notations
# N: batch size
# C: 2 or 3 (channel dimension/#coordinates)
# T: #frames
# J: #joints
# E: #entities (2 if hands are separated)


class HF_PoseRGB(nn.Module):
    def __init__(self, microaction_window_size, num_joints, num_classes, num_verbs, num_nouns,
                 embedding_dim_final=256, use_2d_pose=False, dropout=0,
                 trajectory_atten_dim_per_head=4, trajectory_tcn_kernel_size=3, trajectory_tcn_stride=[1,2,2], trajectory_tcn_dilations=[1,2],
                 use_global_wrist_reference=True, include_orientation_in_global_wrist_ref=True, use_both_wrists=True, separate_hands=True,
                 tf_heads=8, tf_layers=2,
                 rgb_input_feat_dim=2048, MIB_block=True, modality='both', rgb_frames_to_use=-1 # -1: Use 1 rgb per microaction for all microactions
                 ):
        super().__init__()

        """
        Differences with HF_Pose:
        (-) microaction_overlap: Removed as for multimodal, we assume non-overlapping microactions.
        (+) num_verbs and num_nouns: #verb and #noun classes for the dataset along with num_classes (#actions).
        (+) rgb_input_feat_dim: Added to take the input feature dimension of the RGB stream.
        (+) MIB_block: Added to enable/disable the Modal Interaction Block.
        (+) modality: Added to specify the modality to be used in the transformer. pose/rgb/both
        (+) rgb_frames_to_use: Added to specify the number of RGB frames to use per microaction. 
                               Default: -1 (Use 1 RGB per microaction for all microactions). Other supported cases: 1,2,4.
        """

        self.MIB_block = MIB_block
        
        self.seg_len = microaction_window_size

        self.pose_net = HF_Pose(microaction_window_size, num_joints, num_classes, 
                 embedding_dim_final, use_2d_pose, dropout, 0.0, # microaction_overlap=0,
                 trajectory_atten_dim_per_head, trajectory_tcn_kernel_size, trajectory_tcn_stride, trajectory_tcn_dilations,
                 use_global_wrist_reference, include_orientation_in_global_wrist_ref, use_both_wrists, separate_hands,
                 tf_heads, tf_layers)
        # Disable the transformer and classifier in the pose_net. Only microaction encoding will be used.
        self.pose_net.pose_tf = nn.Identity()
        self.pose_net.classifier = nn.Identity()

        # Linear projections before concatenation
        self.rgb_proj = nn.Linear(rgb_input_feat_dim, embedding_dim_final)
        self.pose_proj = nn.Linear(embedding_dim_final, embedding_dim_final)

        # Modality Interaction Block (MIB) MLPs
        self.res_pose_proj = MLP(embedding_dim_final, [embedding_dim_final], dropout=dropout, add_bn_layer=True) # Residual Connection for Pose Tokens
        self.res_rgb_proj = MLP(embedding_dim_final, [embedding_dim_final], dropout=dropout, add_bn_layer=True) # Residual Connection for RGB Tokens
        self.posergb_proj = MLP(embedding_dim_final*2, [embedding_dim_final*2], dropout=dropout, add_bn_layer=True) # Feaute Mixing
        
        self.feat_anticipation_mlp = nn.Sequential(
            MLP(512, [512, 1024], dropout=0.8, add_bn_layer=True),
            nn.Linear(1024,rgb_input_feat_dim)
        )

        self.posergb_tf = Bimodal_TF(embedding_dim_final, num_heads_=tf_heads, num_layers_=tf_layers, dropout=dropout, return_all_tokens=False,
                                     modality=modality, rgb_frames_to_use=rgb_frames_to_use
                                    )

        # Classifiers
        self.classifier = nn.Linear(embedding_dim_final, num_classes)
        self.classifier_verb = nn.Linear(embedding_dim_final, num_verbs)
        self.classifier_noun = nn.Linear(embedding_dim_final, num_nouns)
        
    def forward(self, x, x_rgb, return_atten_map=False):
        # Get pose features for all microactions
        maction_pose_tokens = self.pose_net(x) # (B, T, 256), pose tokens for all microactions
        pose_maction_feat = self.pose_proj(maction_pose_tokens) # (B, T, 256)
        # Get RGB features for all microactions
        B,T,_ = x_rgb.shape
        x_rgb = F.normalize(x_rgb,dim=-1) # Normalize full and/or cropped image features.
        rgb_maction_feat = self.rgb_proj(x_rgb.view(B*T,-1)).view(B,T,-1)
        
        # Normalize and concatenate pose and RGB features
        pose_maction_feat = F.normalize(pose_maction_feat, dim=-1)
        rgb_maction_feat = F.normalize(rgb_maction_feat, dim=-1)
        posergb_maction_feat = torch.cat([pose_maction_feat, rgb_maction_feat], dim=-1) # (B, T, 512). NOTE: Pose first, RGB second.
        
        # Pass all features through MIB MLPs
        pose_maction_feat = self.res_pose_proj(pose_maction_feat.view(B*T,-1)).view(B,T,-1)
        rgb_maction_feat = self.res_rgb_proj(rgb_maction_feat.view(B*T,-1)).view(B,T,-1)
        posergb_maction_feat = self.posergb_proj(posergb_maction_feat.view(B*T,-1)).view(B,T,-1) # posergb_maction_feat is later used for feature anticipation.
        
        ### Detour for feature anticipation ###
        # Calculate l1 loss for feature anticipation
        rgb_feat_for_next_maction = self.feat_anticipation_mlp(posergb_maction_feat.view(B*T,-1))
        rgb_feat_for_next_maction = rgb_feat_for_next_maction.view(B,T,-1)
        # Reconstructed vs original RGB features
        reconst_rgb_feat = rgb_feat_for_next_maction[:, 0:T-1, :]
        original_rgb_feat = x_rgb[:, 1:T, :] # Shifted by 1 frame as these are the frames we anticipated.
        # Anticipation loss -- l1 or mse
        l1_loss = F.l1_loss(reconst_rgb_feat, original_rgb_feat)
        # l1_loss = F.mse_loss(reconst_rgb_feat, original_rgb_feat)
        ### Detour ends ###

        if self.MIB_block: # Pass mixed features to transformer -- along with residual connections from pose and rgb features
            maction_feat_for_tf =  posergb_maction_feat + torch.cat([pose_maction_feat, rgb_maction_feat], dim=-1)
        else:
            maction_feat_for_tf =  torch.cat([pose_maction_feat, rgb_maction_feat], dim=-1) # Ignored mixed features (no PoseRGB feat)
       
    
        if return_atten_map:
            summary, summary_v, summary_n, atten_map = self.posergb_tf(maction_feat_for_tf, return_atten_map)
        else:
            summary, summary_v, summary_n = self.posergb_tf(maction_feat_for_tf)                
        out = self.classifier(summary)
        out_verb = self.classifier_verb(summary_v)
        out_noun = self.classifier_noun(summary_n)
        
        if return_atten_map:
            return out, out_verb, out_noun, l1_loss.unsqueeze(0), atten_map
        else:
            return out, out_verb, out_noun, l1_loss.unsqueeze(0)

        
if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')

    TOTAL_FRAMES = 120
    rgb_input_feat_dim= 1536 #  2048  # 
    window_size = 15

    model = HF_PoseRGB(
                    microaction_window_size=window_size, num_joints=21, num_classes=1380, num_verbs=24, num_nouns=90,
                    embedding_dim_final=256, use_2d_pose=False, dropout=0,
                    trajectory_atten_dim_per_head=4, trajectory_tcn_kernel_size=3, trajectory_tcn_stride=[1,2,2], trajectory_tcn_dilations=[1,2],
                    use_global_wrist_reference=True, include_orientation_in_global_wrist_ref=True, use_both_wrists=True, separate_hands=True,
                    tf_heads=8, tf_layers=2,
                    rgb_input_feat_dim=rgb_input_feat_dim, MIB_block=True,
                    modality='both', rgb_frames_to_use=-1 # -1: Use 1 rgb per microaction for all microactions
                ).cuda()


    N, C, T, V, M = 1, 3, TOTAL_FRAMES, 21, 2

    x = torch.randn(N,C,T,V,M).cuda()
    x_rgb = torch.randn(N, (TOTAL_FRAMES//window_size), rgb_input_feat_dim).cuda()

    out, out_v, out_n, feat_ant_loss =model(x, x_rgb, return_atten_map=False)

    print(out.shape)
    print(out_v.shape)
    print(out_n.shape)
    print(feat_ant_loss.shape)    

    print('Model total # params:', count_params(model))


    ### Efficiency metrics

    flops, params = profile(model, inputs=(x,x_rgb))

    print(f"FLOPs: {flops / 1e9} GFLOPs")
    print("#param: ", params)


    num_samples = 100  # Adjust as needed
    total_time = 0

    for _ in tqdm(range(num_samples)):
        start_time = time.time()
        with torch.no_grad():
            _ = model(x,x_rgb)
        end_time = time.time()
        total_time += end_time - start_time

    average_inference_time = total_time / num_samples
    print(f"Average Inference Time: {average_inference_time} seconds")

    gflops = flops / (average_inference_time * 1e9)
    print(f"GFLOPS: {gflops} GFLOPs/s")
    
    gflop_dict, _ = flop_count(model, (x,x_rgb))
    gflops = sum(gflop_dict.values())
    print("GFLOPs: ", gflops)



