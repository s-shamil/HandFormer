import numpy as np
import torch
import torch.nn.functional as F

def _crop_indices_subsegment(total_frames, p_interval, fps=60):
    # Partially adopted from ISTA-Net (https://github.com/Necolizer/ISTA-Net/tree/main)
    begin = 0
    end = total_frames
    if len(p_interval) == 1: # Test or Validation. No stochasticity
        p = p_interval[0]
        bias = int((1-p) * total_frames/2)
        selected_indices = [i for i in range(begin+bias, end-bias)]
    else:
        # Possibly uneven trimming from both ends
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(total_frames*p)),60), total_frames) # constraint cropped_length lower bound as 60 if total>60
        bias = np.random.randint(0,total_frames-cropped_length+1)
        selected_indices = [i for i in range(begin+bias, begin+bias+cropped_length)]

        # # Even trimming from both ends.
        # p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        # bias = int((1-p) * total_frames/2)
        # selected_indices = [i for i in range(begin+bias, end-bias)]

    if fps != 60:
        increm = 60//fps
        selected_indices = [selected_indices[i] for i in range(0,len(selected_indices),increm)] 
        
    return np.array(selected_indices)

def _resize_data_temporal(data_numpy, window_size):
    # Partially adopted from ISTA-Net (https://github.com/Necolizer/ISTA-Net/tree/main)
    # T -> len(selected_indices_pose)
    C,T,V,M = data_numpy.shape
    valid_size = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0) # =T, as we have not done any zero padding until now
    data = data_numpy[:, 0:valid_size, :, :]

    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, valid_size)
    data = data[None, None, :, :] # makes the shape (1,1,CVM, valid_size). 
    # Interpolation-->The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
    data = F.interpolate(data, size=(C * V * M, window_size), mode='bilinear',align_corners=False).squeeze() # up sample or down sample
    data = data.contiguous().view(C, V, M, window_size).permute(0, 3, 1, 2).contiguous().numpy()

    return data