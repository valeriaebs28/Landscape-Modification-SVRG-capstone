# SVRG landscape modification 
from torch.optim import Optimizer
import copy
import torch

def lm_function(x):
    return x

c = 2
eps = 1
func = lm_function
# if available, we want to operate on a gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SVRGLM_k(Optimizer):
    r"""Optimization class for calculating the gradient of one iteration.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: SVRG-LM")
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SVRGLM_k, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

    def set_u(self, new_u):
        """Set the mean gradient for the current epoch. 
        """
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):  
            for u, new_u in zip(u_group['params'], new_group['params']):
                # modified to include the grad-lm variation
                u.grad = new_u.grad.clone()
                # grad = u.grad.data
                # u.grad.data = u.grad.data/lm_function(new_u.data.clone()) #added
                # u.grad.data = grad/(func(torch.max(torch.zeros(new_u.data.size(), device = device), running_loss - c)) + eps)
                
    def step(self, params):
        """Performs a single optimization step.
        """
        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            weight_decay = group['weight_decay']

            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                # core SVRG gradient update 
                # check if this line needs to be changed (only)
                # new_d = p.grad.data - q.grad.data + u.grad.data
                # new_d = p.grad.data - q.grad.data + u.grad.data

                new_d_lm =  p.grad.data/(func(torch.max(torch.zeros(p.data.size(), device = device), p.data - c)) + eps) - q.grad.data/(func(torch.max(torch.zeros(q.data.size(), device = device), q.data - c)) + eps) + u.grad.data/(func(torch.max(torch.zeros(u.data.size(), device = device), u.data - c)) + eps)
                # print("type of new_d_lm: ", type(new_d_lm))
                # print("new_d_lm: ", new_d_lm)
                # print("type of new_d: ", type(new_d))
                # print("new_d has type: ", type(new_d), "value: ", new_d)
                if weight_decay != 0:
                    new_d_lm.add_(weight_decay, p.data) #should this be changed? to p/f()+eps
                p.data.add_(-group['lr'], new_d_lm)

                # if weight_decay != 0:
                #     new_d.add_(weight_decay, p.data)
                # p.data.add_(-group['lr'], new_d)


class SVRGLM_Snapshot(Optimizer):
    r"""Optimization class for calculating the mean gradient (snapshot) of all samples.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SVRGLM_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """Copies the parameters from another optimizer. 
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]