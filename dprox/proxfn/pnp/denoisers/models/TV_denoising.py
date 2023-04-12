import numpy as np
import torch

def TV_denoising(y0, lamda, iteration=100):
    device = y0.device
    w, h, b  = y0.shape
    zh = torch.zeros([w, h-1, b], device=device, dtype=torch.float32)
    zv = torch.zeros([w-1, h, b], device=device, dtype=torch.float32)
    alpha = 5
    for it in range(iteration):
        x0h = y0 - dht_3d(zh)
        x0v = y0 - dvt_3d(zv)
        x0 = (x0h + x0v) / 2
        zh = clip(zh + 1/alpha*dh(x0), lamda/2)
        zv = clip(zv + 1/alpha*dv(x0), lamda/2)
    return x0

def TV_denoising3d(y0, lamda, iteration=100):
    device = y0.device
    # z = torch.zeros(y0.shape - [1, 1, 1], device=device, dtype=torch.float32)
    w, h, b  = y0.shape
    zh = torch.zeros([w, h-1, b], device=device, dtype=torch.float32)
    zv = torch.zeros([w-1, h, b], device=device, dtype=torch.float32)
    zt = torch.zeros([w, h, b-1], device=device, dtype=torch.float32)
    alpha = 5
    for it in range(iteration):
        x0h = y0 - dht_3d(zh)
        x0v = y0 - dvt_3d(zv)
        x0t = y0 - dtt_3d(zt)
        x0 = (x0h + x0v + x0t) / 3
        zh = clip(zh + 1/alpha*dh(x0), lamda/2)
        zv = clip(zv + 1/alpha*dv(x0), lamda/2)
        zt = clip(zt + 1/alpha*dt(x0), lamda/2)
    return x0

def clip(x, thres):
    return torch.clamp(x, min=-thres, max=thres)

def dht_3d(x):
    return torch.cat([-x[:,0:1,:], x[:,:-1,:]-x[:,1:,:], x[:,-1:,:]], 1)

def dvt_3d(x):
    return torch.cat([-x[0:1,:,:], x[:-1,:,:]-x[1:,:,:], x[-1:,:,:]], 0)

def dtt_3d(x):
    return torch.cat([-x[:,:,0:1], x[:,:,:-1]-x[:,:,1:], x[:,:,-1:]], 2)

def dh(x):
    return x[:,1:,:]-x[:,:-1,:]

def dv(x):
    return x[1:,:,:] - x[:-1,:,:]

def dt(x):
    return x[:,:,1:] - x[:,:,:-1]