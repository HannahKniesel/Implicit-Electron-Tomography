import torch

########################################
# Implicit Model for Reconstruction    #
########################################


class EM_Simulator(torch.nn.Module):

    def __init__(self, num_pos_enc, large, scale = 7, features = 256):
        super(EM_Simulator, self).__init__()
        self.large = large
        self.scale = scale
        
        channels = features
        input_pos_enc = num_pos_enc*2*3 
        
        if(self.large):
            self.layers_1 = torch.nn.Sequential(
                *[
                    torch.nn.Linear(input_pos_enc, channels), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                ])    
            self.layers_2 = torch.nn.Sequential(
                *[
                    torch.nn.Linear(channels + input_pos_enc, channels), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, 1),
                    torch.nn.Sigmoid(),
                ])        
        else: 
            self.layers_1 = torch.nn.Sequential(
                *[
                    torch.nn.Linear(input_pos_enc, channels), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(channels, 1),
                    torch.nn.Sigmoid(),
                ])   

        

    def forward_net(self, x):
        x = x[:,3:]
        y = self.layers_1(x)
        if(self.large):
            return self.layers_2(torch.cat((x, y), axis=-1))
        else:
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

        bool_arr = (x>1) | (x<-1)
        bool_arr = torch.sum(bool_arr, dim=-1)
        bool_arr = bool_arr >=1 
        densities = torch.where(bool_arr, torch.zeros(bool_arr.shape).cuda(), torch.squeeze(densities))

        densities = densities.reshape((bs, num_samples, 1)) 

        if(tv_1d):
            tv_1d_reg = torch.mean(torch.abs(densities[:, :-1, :] - densities[:, 1:, :]))

        densities = torch.sum(densities*distances, dim = 1) 
        densities = torch.exp(-1*densities) #EM Simulation (bs,1)

        if(get_single_densities):
            if(tv_1d):
                return densities, densities_net, tv_1d_reg 
            else: 
                return densities, densities_net 
        
        if(tv_1d):
            return densities, tv_1d_reg

        return densities
