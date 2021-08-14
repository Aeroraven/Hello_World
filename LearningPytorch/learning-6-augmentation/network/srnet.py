import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import os
from functools import partial
import torch.nn as nn

class SRNet(torch.nn.Module):
    def __init__(self, encoder_name, encoder_weights, classes, activation=None):
        super(SRNet, self).__init__()
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=len(classes) if isinstance(classes, list) else classes,
            activation=activation,
        ).encoder
        
        self.avg = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = torch.nn.Flatten()
        if encoder_name == 'resnet34':
            self.fc = torch.nn.Linear(in_features=512, out_features=classes, bias=True)
            self.dropout = torch.nn.Dropout(0.3)
        else:
            raise ValueError(f'encoder name not included')
        self.activation = torch.nn.Softmax()
        
        self.features_forward = {}
        self.features_backward = {}
        self.hooks_up = False
        self.extract_layers = ['encoder.layer4.2.bn2']
        
        self.register_hooks()
        return
    
    def add_forward_hooks(self, _module, _forward_input, forward_output, key):
        if self.hooks_up:
            self.features_forward[key] = forward_output
            self.cam_valid = False
        return

    def add_backward_hooks(self, _module, _backward_input, backward_output, key):
        if self.hooks_up:
            self.features_backward[key] = backward_output
            self.cam_valid = False
        return
    
    def register_hooks(self):
        """register hooks for extracted features, when hooks_up is True
        """
        for name, module in self.named_modules():
            if name in self.extract_layers:
                module.register_forward_hook(partial(self.add_forward_hooks, key=name))
                module.register_backward_hook(partial(self.add_backward_hooks, key=name))
        return

    def compute_cam(self, num_class, cam_shape=(320, 320), key='encoder.layer4.2.bn2'):
        """compute cam of a specific layer
        :key the layer name
        """       
        cam_stack = []
        feature_map = self.features_forward[key]
        for idx in range(num_class):
            bt, ch, w, h = feature_map.size()
            weights = self.fc.weight[idx,:].expand(bt, w, h, ch).permute(0,3,1,2) ### expand
            temp = weights * feature_map ### dot product
            cam = torch.sum(temp, dim=1).unsqueeze(1) ### weighted sum
            ### upsample
            cam = F.interpolate(cam, cam_shape, mode='bilinear', align_corners=False)
            ### minmax norm along batchsize
            bt, ch, w, h = cam.size()
            cam = cam.view(bt, -1)
            cam_min = torch.min(cam, dim=1, keepdim=True)[0]
            cam_max = torch.max(cam, dim=1, keepdim=True)[0]
            cam = (cam - cam_min)/ (torch.where(cam_max != 0, cam_max, torch.ones_like(cam_max))-cam_min)
            cam = cam.view(bt, ch, w, h).squeeze(1)
            cam_stack.append(cam)
        cam_stack = torch.stack(cam_stack, dim=1)
        
        return cam_stack
    
    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.avg(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x

        
def load_encoder_weights(unet, root, experiment='ss-patch20-rotate', model='model', partial=1000, verbose=False):
    
    backbone = torch.load(os.path.join(root, 'results/'+experiment+'/'+model+'.pth'))
    backbone_state_dict = backbone.encoder.state_dict()
    state = unet.state_dict()
    keys = [k for k, _ in state.items()]
    count_layer = 0
    for k in keys:
        if k in backbone.state_dict():
            state[k] = backbone.state_dict()[k]
            count_layer += 1
            if verbose:
                print(f'transfer {k}')
            if count_layer>partial:
                break

    unet.load_state_dict(state)
    return unet

def encoder_freeze_thaw(unet, enabled=False, num=4):
    m = list(unet.encoder.children())
    for x in m[:min(num,len(m))]:
        for p in x.parameters():
            p.requires_grad = enabled
            
    return

def load_srnet_weights(srnet, root, experiment='ss-patch20-rotate', model='model', verbose=False):
    backbone = torch.load(os.path.join(root, 'results/'+experiment+'/'+model+'.pth'))
    backbone_state_dict = backbone.state_dict()
    state = srnet.state_dict()
    keys = state.keys()
    for k in keys:
        if k in backbone_state_dict:
            state[k] = backbone_state_dict[k]
            if verbose:
                print(f'transfer {k}')

    srnet.load_state_dict(state)
    return srnet

