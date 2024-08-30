import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trajectory_SelfAttention(nn.Module):
    def __init__(self, input_channels, temporal_length, N, num_heads, dropout_rate=0.2):
        super(Trajectory_SelfAttention, self).__init__()
        self.num_heads = num_heads                      # Number of attention heads
        self.input_channels = input_channels            # Feature vector for each joint at each time step of the trajectory
        self.temporal_length = temporal_length          # Length of the trajectory
        self.N = N                                      # Number of trajectories (tokens)
        D = input_channels                              # Alias for input_channels
        H = num_heads                                   # Alias for num_heads
        d_k = D // H                                    # Dimensionality per head
        self.depth = d_k                                # Depth or dimensionality of each head
        self.scale = (d_k*temporal_length)**-0.5        # Scaling factor for attention weights

        # QKV linear projections
        self.query = nn.Linear(D, H * d_k)
        self.key = nn.Linear(D, H * d_k)
        self.value = nn.Linear(D, H * d_k)
        # Output projection
        self.final_linear = nn.Linear(H * d_k, D)

        self.dropout0 = nn.Dropout(dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm([N, temporal_length, D])
        self.norm2 = nn.LayerNorm([N, temporal_length, D])

        # Learnable attention map adjustment
        self.att_map_adjuster = nn.init.uniform_(nn.Parameter(torch.Tensor(1, num_heads, N, N), requires_grad=True), -1e-6, 1e-6)
        # Learnable head weights
        self.head_weight = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)

        # FFN
        self.feed_forward = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(D * 4, D),
        )

        # [Optional] Positional Encoding. Indicates the particular joint id for each token. Not temporal. 
        self.positional_encoding = nn.Parameter(torch.randn(1, N, 1, D), requires_grad=True)


    def split_heads(self, x, batch_size):
        # Input shape is (B, N, T, C)
        x = x.view(batch_size, self.N, self.temporal_length, self.num_heads, self.depth) 
        # NOTE: ^ Feature dimension (C) is factorized into heads*depth, remember the order when merging them back.
        return x.permute(0, 3, 1, 2, 4).contiguous() # B, H, N, T, d_k

    def forward(self, x):
        # input x--> BCTN (Batch, C, T, Tokens) ... #Tokens may vary based on the number of joints and/or global reference token
        # post-norm forward
        batch_size = x.size(0)
        x = x.permute(0, 3, 2, 1).contiguous()  # BCTN -> BNTC
        pe = self.positional_encoding.repeat(batch_size, 1, self.temporal_length, 1) # Repeat positional encoding for each batch and for each time step
        
        x_with_pe = x + pe # Add positional encoding
        qk = x_with_pe

        Q = self.query(qk)
        K = self.key(qk)
        V = self.value(x)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Flatten temporal dimension with feature dimension for Q and K
        B,H,J,_,_ = Q.size() # B, H, N, T, d_k
        Q = Q.view(B,H,J,-1)
        K = K.view(B,H,J,-1)
            
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1)) * self.scale 
        attention_weights = F.softmax(matmul_qk, dim=-1) # [B,H,N,N] -- H heads each with NxN attention maps
        attention_weights = attention_weights * self.head_weight # Adjust by learnable head weights
        attention_weights = attention_weights + self.att_map_adjuster.to(attention_weights.dtype).repeat(batch_size, 1, 1, 1) # Add learnable attention map adjustment        
        attention_weights = self.dropout0(attention_weights) # Apply dropout

        # Reshape the value tensor to collapse the last two dimensions into a single dimension
        V_reshaped = V.view(batch_size, self.num_heads, self.N, -1)  # Shape: [B, H, N, T*d_k]
        attended_features = torch.matmul(attention_weights, V_reshaped)  # (B, H, N, N) x (B, H, N, T*d_k) -> B, H, N, T*d_k

        # Reshape attended_features back to shape BNTC
        attended_features = attended_features.view(batch_size, self.num_heads, self.N, self.temporal_length, -1)  # Shape: [B, H, N, T, d_k]
        attended_features = attended_features.permute(0,1,4,2,3).contiguous() # B, H, d_k, N, T
        attended_features = attended_features.view(batch_size, self.num_heads*self.depth, self.N, self.temporal_length) # B, H*d_k, N, T == B, C, N, T
        attended_features = attended_features.permute(0,2,3,1).contiguous() # B, N, T, C
        attended_features = self.final_linear(attended_features) # B, N, T, C

        # FFN and Norm
        x2 = self.dropout1(attended_features)
        x = x + x2
        x = self.norm1(x)
        x2 = self.feed_forward(x)
        x2 = self.dropout2(x2)
        output = x + x2
        output = output.permute(0, 3, 2, 1).contiguous() # Back to BCTN

        return output

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    # Example usage
    batch_size, input_channels, temporal_length, N, num_heads = 16, 64, 15, 12, 4
    x = torch.randn(batch_size, input_channels, temporal_length, N)
    model = Trajectory_SelfAttention(input_channels, temporal_length, N, num_heads)
    output = model(x)
    print("Output shape:", output.shape)
    # print parameter count
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))