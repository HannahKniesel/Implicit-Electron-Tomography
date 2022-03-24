import torch
from torch.distributions import Normal 

########################################
# Normalizing Flow for Noise Modelling #
########################################

# Layers are implemented after: 
# REZENDE, Danilo; MOHAMED, Shakir. Variational inference with normalizing flows. In: International conference on machine learning. PMLR, 2015. S. 1530-1538.

class Radial1DFlow(torch.nn.Module):
    
    def __init__(self, 
                 alpha = None,
                 beta = None,
                 z_zero= None):
        super(Radial1DFlow, self).__init__()

        if z_zero is None:
            self.z_zero = torch.nn.Parameter(torch.zeros((1,)))
            self.z_zero.data.uniform_(-1., 1.)
        else:
            self.z_zero = z_zero

        if alpha is None:
            self.alpha = torch.nn.Parameter(torch.zeros((1,)))
            self.alpha.data.uniform_(-1., 1.)
        else:
            self.alpha = alpha

        if beta is None:
            self.beta = torch.nn.Parameter(torch.zeros((1,)))
            self.beta.data.uniform_(-1., 1.)
        else:
            self.beta = beta


    def forward(self, zk):
        alpha_t = torch.exp(self.alpha)
        beta_t = torch.log(1+torch.exp(self.beta)) - alpha_t
        diff = zk - self.z_zero
        addition = (beta_t*diff)/(alpha_t + torch.abs(diff))
        if torch.any(torch.isnan(zk + addition))\
         or torch.any(torch.isinf(zk + addition)):
            print(self.alpha, self.beta)
        return zk + addition


    def inverse(self, y):
        alpha_t = torch.exp(self.alpha)
        beta_t = torch.log(1+torch.exp(self.beta)) - alpha_t

        a_pos = -1.
        a_neg = 1.
        b_pos = y - alpha_t + self.z_zero - beta_t
        b_neg = - y - alpha_t - self.z_zero - beta_t
        c_pos = y*alpha_t - y*self.z_zero + beta_t*self.z_zero
        c_neg = y*alpha_t + y*self.z_zero + beta_t*self.z_zero

        sqrt_val_pos = b_pos*b_pos - 4*a_pos*c_pos
        sqrt_val_neg = b_neg*b_neg - 4*a_neg*c_neg

        z_1 = (-b_pos + torch.sqrt(sqrt_val_pos))/(2.*a_pos)
        z_2 = (-b_pos - torch.sqrt(sqrt_val_pos))/(2.*a_pos)
        z_3 = (-b_neg + torch.sqrt(sqrt_val_neg))/(2.*a_neg)
        z_4 = (-b_neg - torch.sqrt(sqrt_val_neg))/(2.*a_neg)

        z_zeros = torch.zeros_like(z_1, dtype=torch.float32)
        z_pos =  torch.where(z_1 > self.z_zero, z_1, z_zeros)
        z_pos =  z_pos + torch.where(z_2 > self.z_zero, z_2, z_zeros)
        z_neg = torch.where(z_3 <= self.z_zero, z_3, z_zeros)
        z_neg = z_neg + torch.where(z_4 <= self.z_zero, z_4, z_zeros)
        
        return z_pos+z_neg
            

    def forward_log_det_jacobian(self, zk):
        alpha_t = torch.exp(self.alpha)
        beta_t = torch.log(1+torch.exp(self.beta)) - alpha_t
        
        r = torch.abs(zk - self.z_zero)
        tmp = 1 + beta_t * (1./(alpha_t + r))
        tmp2 = -1./(alpha_t + r)**2
        det = (tmp + beta_t * tmp2 * r) 

        if torch.any(torch.isnan(torch.log(det+1e-8)))\
         or torch.any(torch.isinf(torch.log(det+1e-8))):
            print(self.alpha, self.beta)
            
        return torch.log(det+1e-8)

    def inverse_log_det_jacobian(self, zk):
        return -self.forward_log_det_jacobian(self.inverse(zk))


class CondConvRadial1DFlow(torch.nn.Module):
    
    def __init__(self, num_hidden=8, additional_hidden=False):
        super(CondConvRadial1DFlow, self).__init__()  

        if(additional_hidden):
            self.layers = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(1, num_hidden, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(1, num_hidden, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(num_hidden, 3, 3)
            ]) 
        else: 
            self.layers = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(1, num_hidden, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(num_hidden, 3, 3)
            ]) 
        
    def set_condition_image(self, img):
        self.img = img

    def compute_conditioned_vars(self):
        trans_img = torch.transpose(self.img, 1, -1)
        trans_img = torch.nn.ZeroPad2d(2)(trans_img)
        vars_vals = self.layers(trans_img)
        z_zeros = torch.reshape(vars_vals[:,0,:,:], [-1])
        alphas = torch.reshape(vars_vals[:,1,:,:], [-1])
        betas = torch.reshape(vars_vals[:,2,:,:], [-1])
        return z_zeros, alphas, betas

    def forward(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.forward(zk)

    def inverse(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.inverse(zk)
            
    def forward_log_det_jacobian(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.forward_log_det_jacobian(zk)

    def inverse_log_det_jacobian(self, zk):
        return -self.forward_log_det_jacobian(self.inverse(zk))


class CondMLPRadial1DFlow(torch.nn.Module):
    
    def __init__(self, num_hidden=8, additional_hidden=False):
        super(CondMLPRadial1DFlow, self).__init__()  

        if(additional_hidden):
            self.layers = torch.nn.Sequential(
                *[
                    torch.nn.Linear(1, num_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_hidden, num_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_hidden, 3),
                    torch.nn.Tanh()
                ])   
        else:
            self.layers = torch.nn.Sequential(
                *[
                    torch.nn.Linear(1, num_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_hidden, 3),
                    torch.nn.Tanh()
                ])    
        self.scale = 10.0 
        
    def set_condition_image(self, img):
        self.img = img

    def compute_conditioned_vars(self):
        trans_img = torch.reshape(self.img, (-1, 1))
        vars_vals = self.layers(trans_img)*self.scale
        z_zeros = torch.reshape(vars_vals[:,0], [-1])
        alphas = torch.reshape(vars_vals[:,1], [-1])
        betas = torch.reshape(vars_vals[:,2], [-1])
        return z_zeros, alphas, betas

    def forward(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.forward(zk)

    def inverse(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.inverse(zk)
            
    def forward_log_det_jacobian(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.forward_log_det_jacobian(zk)

    def inverse_log_det_jacobian(self, zk):
        return -self.forward_log_det_jacobian(self.inverse(zk))



class CondMLPVarRadial1DFlow(torch.nn.Module):
    
    def __init__(self, num_hidden=8):
        super(CondMLPVarRadial1DFlow, self).__init__()  

        self.layers = torch.nn.Sequential(
            *[
                torch.nn.Linear(1, num_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(num_hidden, 3),
                torch.nn.Tanh()
            ])    
        self.scale = 10.0 
        
    def set_condition_image(self, img):
        self.img = img

    def compute_conditioned_vars(self):
        trans_img = torch.reshape(self.img, (-1, 1))
        vars_vals = self.layers(trans_img)*self.scale
        z_zeros = torch.reshape(vars_vals[:,0], [-1])
        alphas = torch.reshape(vars_vals[:,1], [-1])
        betas = torch.reshape(vars_vals[:,2], [-1])
        return z_zeros, alphas, betas

    def forward(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.forward(zk)

    def inverse(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.inverse(zk)
            
    def forward_log_det_jacobian(self, zk):
        z_zeros, alphas, betas = self.compute_conditioned_vars()
        radial_flow = Radial1DFlow(alphas, betas, z_zeros)
        return radial_flow.forward_log_det_jacobian(zk)

    def inverse_log_det_jacobian(self, zk):
        return -self.forward_log_det_jacobian(self.inverse(zk))
        

class NoiseModel(torch.nn.Module):

    def __init__(self, 
                 num_layers_cond = 8,
                 num_layers = 8,
                 num_hidden_cond = 16,
                 additional_hidden = False,
                 cond_type = "conv"):
        super(NoiseModel, self).__init__()

        # Create the flow conditioned layers.
        self.cond_flow_layers = torch.nn.ModuleList()
        for i in range(num_layers_cond):
            if cond_type == "conv":
              self.cond_flow_layers.append(
                CondConvRadial1DFlow(num_hidden_cond, additional_hidden = additional_hidden))
            elif cond_type == "mlp":
              self.cond_flow_layers.append(
                CondMLPRadial1DFlow(num_hidden_cond, additional_hidden = additional_hidden))              
            
        # Create the flow layers.
        self.flow_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.flow_layers.append(
                Radial1DFlow())

    def log_prob(self, observation, cond_img):

        # Execute the flow.
        log_det = 0
        transf_img = torch.reshape(observation - cond_img, [-1])
        for cur_layer in self.flow_layers:
            transf_img = cur_layer.forward(transf_img)
            log_det = log_det + cur_layer.forward_log_det_jacobian(transf_img)
        for cur_layer in self.cond_flow_layers:
            cur_layer.set_condition_image(cond_img)
            transf_img = cur_layer.forward(transf_img)
            log_det = log_det + cur_layer.forward_log_det_jacobian(transf_img)

        # Get probabilities.
        distribution = Normal(
            torch.zeros_like(torch.reshape(cond_img, [-1])), 
            torch.ones_like(torch.reshape(cond_img, [-1])))

        """ print(torch.max(transf_img))
        print(torch.min(transf_img))"""

        return distribution.log_prob(transf_img) + log_det

    def sample_noise(self, cond_img):

        # Sample initial distribution.
        distribution = Normal(
            torch.zeros_like(torch.reshape(cond_img, [-1])), 
            torch.ones_like(torch.reshape(cond_img, [-1])))
        noise_img = distribution.sample()

        # Execute the flow.
        num_flow_layers = len(self.cond_flow_layers)
        for cur_layer_iter in range(num_flow_layers):
            cur_layer = self.cond_flow_layers[num_flow_layers - cur_layer_iter - 1]
            cur_layer.set_condition_image(cond_img)
            noise_img = cur_layer.inverse(noise_img)

        num_flow_layers = len(self.flow_layers)
        for cur_layer_iter in range(num_flow_layers):
            cur_layer = self.flow_layers[num_flow_layers - cur_layer_iter - 1]
            noise_img = cur_layer.inverse(noise_img)
        

        # Get probabilities.
        return torch.reshape(noise_img, cond_img.shape)
