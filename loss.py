import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from util.utils import cosine_loss
from torch.autograd import Variable
# Support: ['FocalLoss']


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class FCGLoss(nn.Module):
    def __init__(self, features_size, total_classes, device_id, s=64.0, k=80.0, a=0.93, b=1.30):
        super(FCGLoss, self).__init__()
        self.features_size = features_size
        self.total_classes = total_classes
        self.device_id = device_id
        self.s = s
        self.k = k
        self.a = a
        self.b = b
        self.weight = Parameter(torch.FloatTensor(total_classes, features_size)) # [num_class, 512]
        # nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, inputs, labels):
        if self.device_id == None:
            cosine = F.linear(F.normalize(inputs), F.normalize(self.weights))
        else:
            x = inputs
            sub_weight = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weight[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))

            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weight[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        
        cosine = cosine.clamp(-1, 1)
        # --------------------------- s*cos(theta) ---------------------------
        output = cosine * self.s
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        zero_hot = torch.ones(cosine.size())
        if self.device_id != None:
            zero_hot = zero_hot.cuda(self.device_id[0])
        zero_hot.scatter_(1, labels.view(-1, 1), 0)

        WyiX = torch.sum(one_hot * output, 1) # [B]
        with torch.no_grad():
            theta_yi = torch.acos(WyiX / self.s)
            mode_yi = torch.mode(theta_yi)[0]
            weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))
        intra_loss = - weight_yi * WyiX # [B]

        Wj = zero_hot * output # [B, classes]
        with torch.no_grad():
            theta_j = torch.acos(Wj / self.s) # [B, classes]
            weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))
        inter_loss = torch.sum(weight_j * Wj, 1) # [B]
        '''
        with open('f_theta.txt', 'a') as f:
            f.write('mode_theta_yi: {:.3f}, theta_yi {:.3f}, theta_j {:.3f}\n'.format(mode_yi.item(), theta_yi.mean().item(), theta_j.sum()/(theta_j.shape[0]*(theta_j.shape[1]-1))))
        f.close()
        '''
        loss = intra_loss.mean() + inter_loss.mean()
        # print('intra: ', intra_loss.mean(), 'inter: ', inter_loss.mean())
        return loss, self.weight, mode_yi.item()

def cal_loss_p(inputs, labels, f_centers, device_id, total_classes, mode_yi, con_a, con_b):
    s = 64.0
    k = 80.0
    a = con_a
    b = con_b
    # with torch.no_grad():
        # f_centers = Variable(f_centers)
    if device_id == None:
        cosine = F.linear(F.normalize(inputs), F.normalize(f_centers))
    else:
        x = inputs
        sub_weight = torch.chunk(f_centers, len(device_id), dim=0)
        temp_x = x.cuda(device_id[0])
        weight = sub_weight[0].cuda(device_id[0])
        cosine = F.linear(F.normalize(temp_x), F.normalize(weight))

        for i in range(1, len(device_id)):
            temp_x = x.cuda(device_id[i])
            weight = sub_weight[i].cuda(device_id[i])
            cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(device_id[0])), dim=1)
    cosine = cosine.clamp(-1, 1)
    # --------------------------- s*cos(theta) ---------------------------
    output = cosine * s
    one_hot = torch.zeros(cosine.size())
    if device_id != None:
        one_hot = one_hot.cuda(device_id[0])
    one_hot.scatter_(1, labels.view(-1, 1), 1)

    zero_hot = torch.ones(cosine.size())
    if device_id != None:
        zero_hot = zero_hot.cuda(device_id[0])
    zero_hot.scatter_(1, labels.view(-1, 1), 0)

    WyiX = torch.sum(one_hot * output, 1) # [B]
    with torch.no_grad():
        theta_yi = torch.acos(WyiX / s) # [B]
        weight_yi = (1.0 + (theta_yi - mode_yi)) * 1.0 / (1.0 + torch.exp(-k * (theta_yi - a))) # [B]
    intra_loss = - weight_yi * WyiX # [B]

    Wj = zero_hot * output # [B, classes]
    with torch.no_grad():
        theta_j = torch.acos(Wj / s) # [B, classes]
        weight_j = 1.0 / (1.0 + torch.exp(k * (theta_j - b)))
    inter_loss = torch.sum(weight_j * Wj, 1) # [B]
    loss = intra_loss.mean() + inter_loss.mean()
    '''
    with open('p_theta.txt', 'a') as f:
        f.write('theta_yi {:.3f}, theta_j {:.3f}\n'.format(theta_yi.mean().item(), theta_j.sum()/(theta_j.shape[0]*(theta_j.shape[1]-1))))
    f.close()
    '''
    return loss


        
