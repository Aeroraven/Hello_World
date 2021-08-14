import imp
import pickle
import os
import threading
import torch

"""
utility
save data
"""
def save_config(target, source):
    m = imp.load_source("", source)
    objs = dir(m)
    conf = {}
    for k in objs:
        if '__' not in k:
            conf[k] = getattr(m, k)
    
    with open(target, 'wb') as f:
        pickle.dump(conf, f)
    return

def save_code(root, save_root, ipynbname):
    with open(os.path.join(root, ipynbname), 'r') as f:
        text = f.read()
    with open(os.path.join(save_root, ipynbname), 'w') as f:
        f.write(text)
    return

def save_model_async(model, save_root, save_name='model.pth'):
    torch.save(model, os.path.join(save_root, save_name))
    #print('model saved')
    return

def save_model(model, save_root, save_name='model.pth'):
    timer = threading.Timer(1, save_model_async, [model, save_root, save_name])
    timer.start()
    return
