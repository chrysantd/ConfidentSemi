import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable

class GaussianNoise(nn.Module):
    
    def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05,use_gpu=False):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = torch.zeros(self.shape)
        if use_gpu:
            self.noise = self.noise.cuda()
        self.std = std
        
    def forward(self, x):
        self.noise.normal_(0, std=self.std)
        return x + self.noise

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

    
class SoftmaxWeightConv1d(nn.Conv1d):    

    def transform_weight(self):
        return torch.softmax(self.weight, 1)

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)        

    def forward(self, input):
        return nn.functional.conv1d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)

        

class CNN(nn.Module):
    
    def __init__(self, batch_size, std, p=0.5,fm0 = 1, fm1=16, fm2=32, num_classes=1,use_gpu=False):
        super(CNN, self).__init__()
        self.fm0   = fm0 
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.gn    = GaussianNoise(batch_size,input_shape=(self.fm0,28,28), std=self.std,use_gpu=self.use_gpu)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(self.fm0, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 7 * 7, self.num_classes)    

    def feat(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        return x
    
    def forward(self, x):
        x = self.fc(self.feat(x))
        return x


class CNN2(nn.Module):
    
    def __init__(self, batch_size, std, p=0.5,fm0 = 1, fm1=16, fm2=32, num_classes=1,use_gpu=False):
        super(CNN2, self).__init__()
        self.fm0   = fm0 
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.gn    = GaussianNoise(batch_size,input_shape=(self.fm0,32,32), std=self.std,use_gpu=self.use_gpu)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(self.fm0, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 8 * 8, self.num_classes)    

    def feat(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 8 * 8)
        x = self.drop(x)
        return x
    
    def forward(self, x):
        x = self.fc(self.feat(x))
        return x