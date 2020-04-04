import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import save_tensor
from torch import optim
from loss_functions import SoftmaxLoss
from loss_functions import ASoftmaxLoss

# Very small value to prevent zeros
micro_delta = 0.00001

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)



class StatisticalPooling(nn.Module):
    def __init__(self, in_features, input_size=392):
        super(StatisticalPooling, self).__init__()

    def forward(self, x):
        x1 = torch.mean(x, dim=1)
        x2 = torch.std(x, dim=1)
        x = torch.cat((x1, x2), dim=1)
        return x, 0


i = 0
class DebugLayer(nn.Module):
    def __init__(self):
        super(DebugLayer, self).__init__()

    def forward(self, x):
        global i
        i = i + 1
        save_tensor(x, "x" + str(i))
        #print(x.size())
        exit(1)
        return x 


class TransposedBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(TransposedBatchNorm1d, self).__init__()
        self.layer = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(num_features=num_features),
            Transpose(1, 2))
    
    def forward(self, x):
        return self.layer(x)


class Attention(nn.Module):
    def __init__(self, features, heads, device, learning):
        super(Attention, self).__init__()
        self.features = features
        self.heads = heads
        self.learning = learning
        self.attention = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(in_channels=features, out_channels=512, kernel_size=1, padding=0, bias=False), 
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            Transpose(1, 2),
            nn.Linear(in_features=512, out_features=heads, bias=False),
            nn.Softmax(dim=1))
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        alpha = self.attention(x)
        e = torch.bmm(x.transpose(2, 1), alpha)
        e_var = torch.sqrt(torch.bmm(x.transpose(2, 1)**2, alpha) - e**2 + micro_delta)
        
        if self.heads > 1 and self.learning:
            penalty = torch.norm(alpha.transpose(1, 2).bmm(alpha) - torch.eye(self.heads, device=self.device) + micro_delta, dim=(1,2), p='fro').mean() * 0.01
        else:
            penalty = 0

        y = torch.cat((e, e_var), dim=1)
        y = y.reshape(batch_size, (self.features*2)*self.heads)
        # features*2 for mean and std_var. If std_var is not used remove *2
        return y, penalty


# takes a tensor in the form (batch_dim, number_of_features_per_time_slice, number_of_time_slickes)
class TdnnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, context, padding, dilation=1):
        super(TdnnLayer, self).__init__()
        self.layer = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=context, dilation=dilation, padding=padding),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU())

    def forward(self, x):
        return self.layer(x)

class Tdnn(nn.Module):
    def __init__(self):
        super(Tdnn, self).__init__()
        self.layer = nn.Sequential(
            Transpose(1, 2),
            TdnnLayer(in_channels=30, out_channels=512, context=5, padding=0),
            TdnnLayer(in_channels=512, out_channels=512, context=3, dilation=2, padding=1),
            TdnnLayer(in_channels=512, out_channels=512, context=3, dilation=3, padding=2),
            Transpose(1, 2),
            nn.Linear(in_features=512, out_features=512),
            TransposedBatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1500),
            TransposedBatchNorm1d(num_features=1500),
            nn.ReLU())

    def forward(self, x):
        return self.layer(x)

class XVectorModel(nn.Module):
    def __init__(self, amount_of_speaker, device, flavour = "stat", learning=True):
        super(XVectorModel, self).__init__()
        self.learning = learning
        self.tdnn = Tdnn()
        self.polling = StatisticalPooling(in_features=512, input_size=392) if flavour == "stat" else Attention(features=1500, heads=5, device=device, learning=self.learning)
        self.fcn6 = nn.Linear(in_features=3000*5 if flavour == "attent" else 3000, out_features=512)
               
        self.post_xvector = nn.Sequential(
                nn.BatchNorm1d(num_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=512),
                nn.BatchNorm1d(num_features=512),
                nn.ReLU())


        self.loss_fn = ASoftmaxLoss(in_features=512, out_features=amount_of_speaker) if flavour == "asoft" else SoftmaxLoss(in_features=512, out_features=amount_of_speaker) 


    def forward(self, x, target = -1):
        x = self.tdnn(x)
        x, penalty = self.polling(x)
        x = self.fcn6(x)
        if self.learning:
            x = self.post_xvector(x)
            x = self.loss_fn(x, target)
            return x, penalty
        else:
            return x

