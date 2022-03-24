import torch


########################################
# Explicit Model for Reconstruction    #
########################################


class EM_GridSimulator(torch.nn.Module):

    def __init__(self, large, grid_scale = 128, scale = 7):
        super(EM_GridSimulator, self).__init__()
        self.scale = scale

        if(large):
            self.grid_scale = grid_scale
        else: 
            self.grid_scale = grid_scale//2

        self.grid =  torch.zeros((self.grid_scale, self.grid_scale, self.grid_scale), device=torch.device("cuda"), requires_grad = True)
       
    def forward_net(self, x):
        y = torch.zeros((x.shape[0],), device = torch.device("cuda"))
        x = (self.grid_scale * ((x+1)/2)).to(torch.long)

        bool_arr = (x>=self.grid_scale) | (x<0)       
        bool_arr = torch.sum(bool_arr, dim=-1)
        bool_arr = bool_arr >=1   

        x = x[~bool_arr,:]
        densities = self.grid[x[:,0], x[:,1], x[:,2]]
        y[~bool_arr] = densities

        return y

       
    def euklid(self, x, y):
        dist = (x-y)**2
        dist = torch.sum(dist, dim = 2)
        dist = torch.sqrt(dist)
        return torch.unsqueeze(dist, dim = -1)

    # x positionally encoded position with shape: (bs, num_samples, positional encoding =input size)
    def forward(self, x,  is_training=True, get_single_densities = False, tv_1d = False):
        if(not is_training):
            return self.forward_net(x)
        bs, num_samples, input_size = x.shape
        
        distances = torch.ones((bs, num_samples, 1)).cuda() 
        distances[:,1:,:] = self.euklid(x[:,:-1,:3], x[:,1:,:3])*self.scale  # distances between samples to discretize integral
        
        x = x.reshape((bs*num_samples, input_size)) 

        densities = self.forward_net(x) #(bs*num_samples, 1)

        if(get_single_densities):
            densities_net = densities.reshape((bs, num_samples, -1)) 
        densities = densities.reshape((bs, num_samples, 1)) 

        if(tv_1d):
            tv_1d_reg = torch.mean(torch.abs(densities[:, :-1, :] - densities[:, 1:, :]))

        densities = torch.sum(densities*distances, dim = 1) #Radon transform/ Line integral: (bs,1)
        densities = torch.exp(-1*densities) #EM Simulation (bs,1)

        if(get_single_densities):
            if(tv_1d):
                return densities, densities_net, tv_1d_reg 
            else: 
                return densities, densities_net 
        
        if(tv_1d):
            return densities, tv_1d_reg

        return densities

