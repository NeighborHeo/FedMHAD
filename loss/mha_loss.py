from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import unittest
import numpy as np
import torch.optim as optim

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class MHALoss(torch.nn.Module):
    '''
    summary : FedMHAD (Multihead Attention loss)
    '''
    def __init__(self, distill_heads = 3):
        super(MHALoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-8)
        self.distill_heads = distill_heads
        self.ssim_loss = SSIM_loss()
        
    def forward(self, client_attentions, central_attention, weight = None):
        '''
        args :
            client_attentions : (num_clients, batch_size, num_heads, img_height, img_width)
            central_attention : (batch_size, num_heads, img_height, img_width)
            weight : (num_clients)
        returns :
            weighted_loss : (1)
        '''
        self.num_clients = client_attentions.shape[0]
        if weight is not None:
            self.weight = weight.clone().detach()
        else:
            self.weight = torch.ones(self.num_clients, dtype=torch.float)
        
        client_attentions = client_attentions[:, :, 0:self.distill_heads, :, :]
        central_attention = central_attention[:, 0:self.distill_heads, :, :]
        # assert len(clients_output) == self.num_clients, "Number of clients_output should match num_clients"
        num_clients = client_attentions.shape[0]
        loss = 0
        for i in range(num_clients):
            img1 = client_attentions[i]
            img2 = central_attention
            ssim_loss = self.ssim_loss(img1, img2)
            loss += self.weight[i] * ssim_loss            
            # cosine_sim = self.cosine_similarity(client_attentions[i].contiguous().view(-1), central_attention.contiguous().view(-1))
            # loss += self.weight[i] * (1 - cosine_sim).mean()
        loss /= num_clients
        # print('MHA Loss : ', loss)
        return loss


class TestMHALoss(unittest.TestCase):
    
    def test_compute_attention_similarity_losses_using_ssim(self):
        n_clients = 5
        batch_size = 16
        n_heads = 3
        img_height = 32
        img_width = 32
        
        # client_attentions = torch.randn(n_clients, batch_size, n_heads, img_height, img_width, requires_grad=True)
        client_attentions = torch.rand(n_clients, batch_size, n_heads, img_height, img_width)
        # central_attention = torch.randn(batch_size, n_heads, img_height, img_width, requires_grad=True)
        central_attention = torch.ones(batch_size, n_heads, img_height, img_width)
        weight = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        ssim_loss = SSIM_loss()
        print(central_attention.size())
        for i in range(n_clients):
            img1 = client_attentions[i]
            img2 = central_attention
            ssim_ = ssim(img1, img2)
            ssim_loss_ = ssim_loss(img1, img2)
            print(ssim_, ssim_loss_)
            
    
    def test_compute_attention_similarity_losses(self):
        n_clients = 5
        batch_size = 16
        n_heads = 3
        img_height = 32
        img_width = 32
        
        client_attentions = torch.randn(n_clients, batch_size, n_heads, img_height, img_width, requires_grad=True)
        central_attention = torch.randn(batch_size, n_heads, img_height, img_width, requires_grad=True)
        weight = [0.2, 0.2, 0.2, 0.2, 0.2]  # Example weight for each client

        cosine_similarity_loss = MHALoss().requires_grad_(True)
        loss = cosine_similarity_loss(client_attentions, central_attention, weight)
        print(loss.grad_fn)
        print(loss.requires_grad)
        print(loss.grad)
        
if __name__ == '__main__':
    unittest.main()