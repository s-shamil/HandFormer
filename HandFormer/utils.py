# This script contains functions from the following repositories:
# https://github.com/Necolizer/ISTA-Net/blob/main/utils/loss.py
# https://github.com/kenziyuliu/MS-G3D/blob/master/utils.py

import torch.nn.functional as F
import torch.nn as nn

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_loss_func(loss_func, loss_args):
    if loss_func == 'LabelSmoothingCrossEntropy':
        loss = LabelSmoothingCrossEntropy(smoothing=loss_args['smoothing'], temperature=loss_args['temperature'])
    elif loss_func == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    else:
        print('Loss Not Included')
        loss = None
    
    return loss

def import_class(name):
    components = name.split('.') # Sample name: models.hf_pose.HandFormer
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)