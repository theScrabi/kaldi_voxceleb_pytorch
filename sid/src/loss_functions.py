import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from utils import save_tensor

class SoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(SoftmaxLoss, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.BatchNorm1d(num_features=out_features)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, target):
        output = self.norm(self.linear(x))
        logs = F.log_softmax(output, dim=1)
        loss = self.cross_entropy(logs, target)
        
        return loss

class ASoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(ASoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        #const
        self.m = 2

        # !!!Important!!! lambda controls how much of psi(theta) influences cos(theta) during learning
        # In the beginning lambda is big, which makes the influence of psi(theta) low,
        # making ASoftmax behave close to regular softmax.
        # During the computation lambda gets smaller making ASoftmax make the influence of m higher
        # max_lambda = 5 however ensures that there is never more then 1/6 psi(theta) influence.
        # This apparently makes asoftmax behave better for bigger datasets.
        # A detailed description for this can be found on the readme of the SpehereFace project:
        # https://github.com/wy1iu/LargeMargin_Softmax_Loss#notes-for-training
        # NOTE: lambda is calculated dynamically for each operation.
        # base, gamma, power, and min control the computation of lambda.
        self.base = 1000
        self.gamma = 0.15
        self.power = 1
        self.min_lambda = 5.0

        self.iter = 0

        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ] 

    def forward(self, x, target):
        
        # the learning rate lambda
        self.iter += 1
        lamb = max(self.min_lambda, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
      
        # alter theta 
        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1, 1)
        theta = cos_theta.acos()
        cos_m_theta = self.mlambda[self.m](cos_theta)

        #psi(theta)
        len_x = x.norm(2, dim=1) 
        k = (self.m * theta / math.pi).floor()
        psi_theta = ((-1.0) ** k) * cos_m_theta - 2*k

        len_x = x.norm(2, dim=1) 

        # prepare target
        one_hot = (cos_theta * 0.0).detach() ## crate a tensor with zeros the same size as cos_theta
        one_hot.scatter_(dim=1, index=target.view(-1, 1), value=1)        

        # combine and add learn parameter
        psi_or_cos_theta = (one_hot * (psi_theta - cos_theta) / (1 + lamb)) + cos_theta
        len_cos_theta = psi_or_cos_theta * len_x.view(-1, 1)

        logs = F.log_softmax(len_cos_theta, dim=1)
        loss = F.cross_entropy(len_cos_theta, target) ### <--- nor sure
        return loss

