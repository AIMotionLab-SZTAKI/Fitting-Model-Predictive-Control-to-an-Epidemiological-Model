from torch import nn
import torch
import numpy as np


class feed_forward_nn(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())
        seq.append(nn.Linear(n_nodes_per_layer,n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0) #bias
    def forward(self,X):
        return self.net(X)

class simple_res_net(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        #linear + non-linear part
        super(simple_res_net,self).__init__()
        self.net_lin = nn.Linear(n_in,n_out)
        self.n_in = n_in
        self.n_out = n_out
        if n_hidden_layers>0:
            self.net_non_lin = feed_forward_nn(n_in,n_out,n_nodes_per_layer=n_nodes_per_layer,n_hidden_layers=n_hidden_layers,activation=activation)
        else:
            self.net_non_lin = None

    def forward(self,x):
        if self.net_non_lin is not None:
            return self.net_lin(x) + self.net_non_lin(x)
        else: #linear
            return self.net_lin(x)


class default_state_net(nn.Module):
    def __init__(self, nx, nu, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_state_net, self).__init__()
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.net = simple_res_net(n_in=nx+np.prod(self.nu,dtype=int), n_out=nx, n_nodes_per_layer=n_nodes_per_layer, \
            n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, x, u):
        net_in = torch.cat([x,u.view(u.shape[0],-1)],axis=1)
        return self.net(net_in)


class default_output_net(nn.Module):
    def __init__(self, nx, ny, nu=-1, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_output_net, self).__init__()
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.feedthrough = nu!=-1
        if self.feedthrough:
            self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
            net_in = nx + np.prod(self.nu, dtype=int)
        else:
            net_in = nx
        self.net = simple_res_net(n_in=net_in, n_out=np.prod(self.ny,dtype=int), n_nodes_per_layer=n_nodes_per_layer, \
            n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, x, u=None):
        feedthrough = self.feedthrough if hasattr(self,'feedthrough') else False
        if feedthrough:
            xu = torch.cat([x,u.view(u.shape[0],-1)],dim=1)
        else:
            xu = x
        return self.net(xu).view(*((x.shape[0],)+self.ny))


class default_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_encoder_net, self).__init__()
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)