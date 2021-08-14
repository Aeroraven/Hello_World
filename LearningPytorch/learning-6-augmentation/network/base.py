
import torch
import os
# from .jsnet import *
import sys
sys.path.append(r'/root/cofs/dissertations/dissertation/models/JigsawPuzzlePytorch-master/')

def load_unet_weights(unet, root, experiment='ss-patch-seg-vary-1', model='model', verbose=False):
    
    backbone = torch.load(os.path.join(root, 'results/'+experiment+'/'+model+'.pth'))
    backbone_state_dict = backbone.state_dict()
    state = unet.state_dict()
    keys = [k for k, _ in state.items()]
    for k in keys:
        if k in backbone.state_dict():
            try:
                if 'segmentation_head' not in k:
                    state[k] = backbone.state_dict()[k]
            except Exception as e:
                print(f'mismatch error = {e}')
            if verbose:
                print(f'transfer {k}')
    
    unet.load_state_dict(state)
    return unet


def load_model_weights(unet, model_path, verbose=False):
    
    backbone = torch.load(model_path)
    backbone_state_dict = backbone.state_dict()
    state = unet.state_dict()
    keys = [k for k, _ in state.items()]
    for k in keys:
        if k in backbone.state_dict():
            try:
                if 'segmentation_head' not in k:
                    state[k] = backbone.state_dict()[k]
            except Exception as e:
                print(f'mismatch error = {e}')
            if verbose:
                print(f'transfer {k}')
    
    unet.load_state_dict(state)
    return unet


def encoder_freeze_thaw_range(unet, enabled=False, s=0, e=1):
    m = list(unet.encoder.children())
    for x in m[s:min(e,len(m))]:
        for p in x.parameters():
            p.requires_grad = enabled
            
    return