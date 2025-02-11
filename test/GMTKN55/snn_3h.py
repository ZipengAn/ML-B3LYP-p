import torch
from torch import nn

mindiv = 1e-10
    
class SNN(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden,
            lamda = 1e-3,
            beta = 1.5,
            use_cuda = True):
        super(SNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden = hidden
        self.lamda = lamda
        self.beta = beta
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden[0]),
            nn.Sigmoid(),
            nn.Linear(self.hidden[0], self.hidden[1]),
            nn.Sigmoid(),
            nn.Linear(self.hidden[1], self.hidden[2]),
            nn.Sigmoid(),
            nn.Linear(self.hidden[2], self.output_dim),
        )

    def forward(self, ml_in):
        inputs = torch.zeros((ml_in.shape[0], self.input_dim), device = self.device)
        rho13 = torch.pow((ml_in[:,0] + ml_in[:,1] + mindiv), 1/3)
        inputs[:,0] = torch.log(rho13)
        inputs[:,1] = torch.log(torch.div(ml_in[:,0] - ml_in[:,1], ml_in[:,1] + ml_in[:,0] + mindiv) + 1)
        inputs[:,2] = torch.log(torch.div(torch.pow((ml_in[:,2] + 2 * ml_in[:,3] + ml_in[:,4]), 0.5), torch.pow(rho13, 4) + mindiv) + 1)

        tau = torch.Tensor(ml_in[:,5] + ml_in[:,6])
        tau_w = torch.div((ml_in[:,2] + 2 * ml_in[:,3] + ml_in[:,4]), (ml_in[:,0] + ml_in[:,1] + mindiv) * 8.0)
        tau_unif = torch.pow(rho13, 5)
        inputs[:,3] = torch.log(torch.div(tau_w, (tau + mindiv)) + 1)
        inputs[:,4] = torch.log(torch.div((tau - tau_w), (tau_unif + mindiv)) + 1)
        inputs[:,5] = torch.log(torch.div(tau, (tau_unif + mindiv)) + 1)
        exc_ = self.model(inputs)
        exc = exc_
        
        return exc
    
def load_param(nw_p, i_dim, hid, w):
    k = 0
    for i in range(hid[0]):
        for j in range(i_dim):
            nw_p['model.0.weight'][i, j] = w[k]
            k += 1
    
    for i in range(hid[0]):
        nw_p['model.0.bias'][i] = w[k]
        k += 1
    
    for i in range(hid[1]):
        for j in range(hid[0]):
            nw_p['model.2.weight'][i, j] = w[k]
            k += 1
    
    for i in range(hid[1]):
        nw_p['model.2.bias'][i] = w[k]
        k += 1
    
    for i in range(hid[2]):
        for j in range(hid[1]):
            nw_p['model.4.weight'][i, j] = w[k]
            k += 1
    
    for i in range(hid[2]):
        nw_p['model.4.bias'][i] = w[k]
        k += 1
    
    for i in range(hid[2]):
        nw_p['model.6.weight'][0, i] = w[k]
        k += 1

    nw_p['model.6.bias'][0] = w[k]
    k += 1

    return nw_p

