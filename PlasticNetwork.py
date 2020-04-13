import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim 

class plasticNet(nn.Module):
  def __init__ (self, neurons):
    super(plasticNet, self).__init__()

    self.neurons = neurons
    self.w = torch.tensor(0.01*torch.randn(neurons, neurons).type(torch.cuda.FloatTensor), requires_grad=True) 
    self.alpha = torch.tensor(.01*torch.randn(neurons, neurons).type(torch.cuda.FloatTensor), requires_grad=True) 
    self.eta = torch.tensor(.01*torch.ones(1).type(torch.cuda.FloatTensor), requires_grad=True) 

  def forward(self, input, yin, hebb):
    clamps = np.zeros(self.neurons)
    zz = torch.nonzero(input.data[0].cpu()).numpy().squeeze() 
    clamps[zz] = 1
    clamps = torch.as_tensor(torch.from_numpy(clamps).type(torch.cuda.FloatTensor).requires_grad_(False))

    yout = torch.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb))) * (1 - clamps) + input * clamps
    hebb =  torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0] * self.eta + (1 - self.eta)*hebb

    return yout, hebb

  def initialZeroState(self):
    return torch.zeros(1, self.neurons).type(torch.cuda.FloatTensor)

  def initialZeroHebb(self):
    return torch.zeros(self.neurons, self.neurons).type(torch.cuda.FloatTensor)