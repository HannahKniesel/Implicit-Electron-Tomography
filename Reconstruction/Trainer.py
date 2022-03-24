import torch
import numpy as np
import logging
import os
from PIL import Image
import traceback
from scipy import ndimage

from Models.Utils_ModelSaveLoad import *
from Utils import *
from Dataloader import Data_Loader

SAVE_STATE = "/save_state/"
FILE_NAME = "training_state.pth"

class Trainer():

    def __init__(self, args, log_folder_path, gt_path):
        self.args = args
        self.log_folder_path = log_folder_path
        self.val_shape_vol = 512

        self.start_iteration = 0

        print("Make path: "+str(log_folder_path+SAVE_STATE))
        os.mkdir(log_folder_path+SAVE_STATE) 

        #resume training
        if(self.args.resume_training):
            try:
                self.ckpt = torch.load(self.args.resume_training)
                self.start_iteration = load_param(0, 'start_iteration', self.ckpt)

                self.args.convergence = load_param(self.args.convergence, 'convergence', self.ckpt)
                self.args.batch_size = load_param(self.args.batch_size, 'batch_size', self.ckpt)
                self.args.accumulate = load_param(self.args.accumulate, 'accumulate', self.ckpt)
                self.args.patch_size = load_param(self.args.patch_size, 'patch_size', self.ckpt)
                self.args.num_validation = load_param(self.args.num_validation, 'num_validation', self.ckpt)

                self.args.pos_enc = load_param(self.args.pos_enc, 'pos_enc', self.ckpt)
                self.args.samples = load_param(self.args.samples, 'samples', self.ckpt)
                self.args.features = load_param(self.args.features, 'features', self.ckpt)

                self.args.nf_cond = load_param(self.args.nf_cond, 'nf_cond', self.ckpt)
                self.args.nf_noncond = load_param(self.args.nf_noncond, 'nf_noncond', self.ckpt)

                self.args.loss = load_param(self.args.loss, 'loss', self.ckpt)
                self.args.tv1d = load_param(self.args.tv, 'tv', self.ckpt)
                self.args.nf_optim = load_param(self.args.nf_optim, 'nf_optim', self.ckpt)
                self.args.nf_lr = load_param(self.args.nf_lr, 'nf_lr', self.ckpt)
                self.args.nf_path = load_param(self.args.nf_lr, 'nf_path', self.ckpt)
                self.args.optim = load_param(self.args.optim, 'optim', self.ckpt)
                self.args.lr = load_param(self.args.lr, 'lr', self.ckpt)

                self.args.grid_res = load_param(self.args.grid_res, 'grid_res', self.ckpt)

                self.args.lr_decay_start = load_param(self.args.lr_decay_start, 'lr_decay_start', self.ckpt)
                self.args.lr_decay_rate = load_param(self.args.lr_decay_rate, 'lr_decay_rate', self.ckpt)

                self.args.defocus = load_param(False, 'defocus', self.ckpt)
                self.args.max_defocus = load_param(0, 'max_defocus', self.ckpt)
                self.args.std = load_param(False, 'std', self.ckpt)
                
                print("Resumed Training. Set args parameters to: "+str(self.args))

            except: 
                self.ckpt = None
                print(traceback.format_exc())
                print("ERROR:: Could not load training from "+str(self.args.resume_training))

        # Load data
        self.data_loader = Data_Loader(self.args.data_path, self.args.num_validation, self.args.pos_enc, gt_path) 
        self.t_positions, self.t_directions, self.t_labels, self.t_angles, self.t_img_width, self.v_positions, self.v_directions, self.v_labels, self.v_angles, self.v_img_width, self.val_files, self.gt_val = self.data_loader.load_data() #(num_imgs, width, height, num_samples, 3) | (num_imgs, width, height, 1)
        
        m = self.t_labels.shape[0]//2
        half = self.t_labels.shape[1]//2
        self.noisy_label = self.t_labels[m-half:m+half,:self.t_labels.shape[0],0]

        if(self.args.defocus):
            max_blur = self.args.defocus
            max_angle = np.max((np.max(self.t_angles), np.max(self.v_angles)))
            self.center_dist = np.absolute(np.linspace(-1, 1, num = self.data_loader.w))
            self.k = (np.tan(np.deg2rad(max_angle))/max_blur)
            self.defocus = self.compute_defocus_images(self.t_angles, self.data_loader.w)
            self.defocus = self.cropdefocus()
            self.t_labels = self.t_labels.astype(np.float32)

        else:
            self.t_positions = self.t_positions.reshape((-1,self.t_positions.shape[-1]))
            self.t_directions = self.t_directions.reshape((-1,self.t_directions.shape[-1]))
            self.t_labels = self.t_labels.reshape((-1,self.t_labels.shape[-1])).astype(np.float32)
        
        print("Shape training data: "+str(self.t_positions.shape))

        # log val_file
        f = open(log_folder_path+"/val_files.txt", "w")
        f.write(str(self.val_files))
        f.close()

        logging.basicConfig(filename=log_folder_path+'/log.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

        try:
            self.gt_vol = open_raw(self.args.gt_vol)
            self.gt_vol = min_max(self.gt_vol[:,:,self.gt_vol.shape[0]//2])
            scale = self.val_shape_vol/self.gt_vol.shape[0]
            self.gt_vol = ndimage.zoom(self.gt_vol, scale)
            self.gt_vol = 1 - np.clip(self.gt_vol, 0, 1)
            self.gt_vol = np.rot90(self.gt_vol, k=3)
        except: 
            print("WARNING::Did not load phantom volume for validation.")
            self.gt_vol = None
        
        return

    def cropdefocus(self):
        num_images = self.defocus.shape[0]/self.data_loader.w
        croped_defocus = np.zeros(self.t_positions.shape[1])
        idx1 = 0
        idx2 = 0

        for i in range(int(num_images)):
            curr_w = self.t_img_width[i]
            start = (self.data_loader.w-curr_w)//2
            end = start + curr_w
            defoc = self.defocus[idx1:idx1+self.data_loader.w]
            croped_defocus[idx2:idx2+curr_w] = defoc[start:end]
            idx1 = idx1+self.data_loader.w
            idx2 = idx2+curr_w
        return croped_defocus

    def get_defocus(self, y, angle):
        alpha = np.squeeze(np.deg2rad(np.absolute(angle)))
        blur_size = (np.tan(alpha)*self.center_dist[y])/self.k
        blur_size = (blur_size).astype(np.int8)
        return blur_size

    def compute_defocus_images(self, angles, img_w):
        defocus = np.zeros((img_w*angles.shape[0]))
        i = 0
        for angle in angles: 
            defocus[i:i+img_w] = self.get_defocus(np.arange(0,self.data_loader.w), angle)
            i += img_w
        return defocus

    def setup_optimizer(self, model_params, optim_type):
        #Optimizer
        if(optim_type=="sgd"):
            optimizer = torch.optim.SGD(model_params, lr=1, momentum=0.9, weight_decay= 1e-3) 
        elif(optim_type=="adam"):
            optimizer = torch.optim.Adam(model_params , lr=1)

        #LRScheduler 
        num_steps = int(self.args.iterations) 
        start_decay = self.args.lr_decay_start 
        decay_steps = num_steps - start_decay
        lr_init = self.args.lr
        lr_end = self.args.lr/self.args.lr_decay_rate
        lr_muls = []
        for i in range(num_steps+1):
            if i > start_decay:
                lr_muls.append(((i-start_decay)/decay_steps)*(lr_end-lr_init) + lr_init)
            else:
                lr_muls.append(lr_init)

        if(self.start_iteration != 0):
            lr_muls = lr_muls[self.start_iteration:]
        lambda_lr = lambda epoch: lr_muls[epoch]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        return optimizer, scheduler


    def save_img(self, path, img):
        img = np.squeeze(img)
        img = np.clip(img, 0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(path)
        return 

    def get_length_by_gaussian(self,defocus):
        kernel_size = defocus*2/self.data_loader.w
        std = 2*self.args.std/self.data_loader.w
        size = np.random.normal(0.0, std)
        out = np.min((size, kernel_size))
        return out

    def get_batch_data_training_rays_jitter(self):
        num_rays = (self.args.patch_size**2)*self.args.batch_size

        random_idx_x = torch.randint(low = 0, high = self.t_positions.shape[0], size=(num_rays,))
        random_idx_y = torch.randint(low = 0, high = self.t_positions.shape[1], size=(num_rays,))

        pos_patches = self.t_positions[random_idx_x,random_idx_y,:].reshape((-1,3))
        dir_patches = self.t_directions[random_idx_x,random_idx_y,:].reshape((-1,3))
        label_patches = self.t_labels[random_idx_x,random_idx_y,:].reshape((-1,1))

        defocus = self.defocus[random_idx_y]        

        for i in range(pos_patches.shape[0]):
            pos_val = pos_patches[i,:]
            dir_val = dir_patches[i,:]

            dir_orth = np.random.uniform(0,100,3)
            idx_replace = np.random.randint(0,3,1)
            idxs = np.arange(3)
            bool_arr = (idxs == idx_replace)
            missing_val = (-np.sum(dir_val[bool_arr]*dir_orth[bool_arr]))/(dir_val[idx_replace]+1e-10)
            dir_orth[idx_replace] = missing_val

            dir_orth = dir_orth/np.linalg.norm(dir_orth)
            scale = self.get_length_by_gaussian(defocus[i])
            pos_patches[i,:] = pos_val+dir_orth*scale

        pos_patches = pos_patches.reshape((self.args.batch_size, self.args.patch_size, self.args.patch_size, 3))
        dir_patches = dir_patches.reshape((self.args.batch_size, self.args.patch_size, self.args.patch_size, 3))
        label_patches = label_patches.reshape((self.args.batch_size, self.args.patch_size, self.args.patch_size, 1))
        return torch.from_numpy(pos_patches).cuda(), torch.from_numpy(dir_patches).cuda(), torch.from_numpy(label_patches).cuda()

 
    def get_batch_data_training_rays(self):
        num_rays = (self.args.patch_size**2)*self.args.batch_size

        random_idx = torch.randint(low = 0, high = self.t_positions.shape[0], size=(num_rays,))
        pos_patches = self.t_positions[random_idx,:].reshape((self.args.batch_size, self.args.patch_size, self.args.patch_size, 3))
        dir_patches = self.t_directions[random_idx,:].reshape((self.args.batch_size, self.args.patch_size, self.args.patch_size, 3))
        label_patches = self.t_labels[random_idx,:].reshape((self.args.batch_size, self.args.patch_size, self.args.patch_size, 1))

        return torch.from_numpy(pos_patches).cuda(), torch.from_numpy(dir_patches).cuda(), torch.from_numpy(label_patches).cuda()

   
    def get_batch_data_validation_img(self, i_img):
        pos_patches = torch.Tensor(self.v_positions[i_img, :, :,:])
        dir_patches = torch.Tensor(self.v_directions[i_img, :, :,:])
        label_patches = torch.Tensor(self.v_labels[i_img, :, :])
        return pos_patches, dir_patches, label_patches

    
    def mle_loss(self, nf, x, y):
        log_prob = nf.log_prob(
            y,                      # noisy projections (labels)
            x)                      # non noisy reconstruction (prediction)
        log_prob = torch.reshape(log_prob, (self.args.batch_size, -1))
        return -torch.mean(log_prob)  

    def validate_nf(self, noise_model): 
        val_clean = np.squeeze(self.gt_val, axis = -1)
        init_shape = val_clean.shape

        val_clean = val_clean.reshape((-1,self.gt_val.shape[-1]))
        val_noisy = np.squeeze(self.v_labels, axis = -1).reshape((-1,self.v_labels.shape[-1]))
        bs = (self.args.patch_size**2)*self.args.batch_size 
        idx = 0
        idx_end = idx+bs
        nll = 0
        i = 0
        while(idx_end < val_noisy.shape[0]):
            noisy = torch.from_numpy((val_noisy[idx:idx_end])).type(torch.FloatTensor).cuda().reshape(self.args.batch_size, self.args.patch_size, self.args.patch_size, 1)
            clean = torch.from_numpy((val_clean[idx:idx_end])).type(torch.FloatTensor).cuda().reshape(self.args.batch_size, self.args.patch_size, self.args.patch_size, 1)
            idx += bs
            idx_end += bs
            i+=1
            nll += self.mle_loss(noise_model, noisy, clean).detach().cpu()
        if(i != 0):
            nll = nll/i
        val_clean = val_clean.reshape(init_shape)
        cuda_val_clean  = torch.from_numpy((val_clean)).type(torch.FloatTensor).cuda().reshape(1, val_clean.shape[0], val_clean.shape[1], 1)
        noise = noise_model.sample_noise(cuda_val_clean)
        noisy_proj = cuda_val_clean + noise
        noisy_proj = torch.squeeze(noisy_proj, axis = -1).detach().cpu()

        return noisy_proj, nll, noise.reshape(self.data_loader.w, -1)


    def validation(self, simulator_model, simulator_approximation):
        samples_coarse = self.args.samples
        samples_fine = 2*samples_coarse

        img_prediction = np.zeros_like(self.v_labels) 
        img_labels = np.zeros_like(self.v_labels) 
        img_prediction_small = np.zeros_like(self.v_labels) 
        w, h, _ = self.v_labels.shape
        x = 0
        y = 0
        break_x_loop = False
        break_y_loop = False
        bs = self.args.patch_size
        while(x < w):
            while(y < h):
                if((x + bs) < w):
                    x_end = x+bs
                else: 
                    x_end = w
                    break_x_loop = True

                if((y + bs) < h):
                    y_end = y+bs
                else: 
                    y_end = h
                    break_y_loop = True

                pos_patches = torch.from_numpy(self.v_positions[x:x_end, y:y_end,:]).cuda()
                dir_patches = torch.from_numpy(self.v_directions[x:x_end, y:y_end,:]).cuda()
                label_patches = torch.from_numpy(self.v_labels[x:x_end, y:y_end,:]).cuda()

                num_patches = 1
                width, height, _ = pos_patches.shape
                pos_patches = pos_patches.reshape((-1,pos_patches.shape[-1]))
                dir_patches = dir_patches.reshape((-1,dir_patches.shape[-1]))
                samples = self.data_loader.sample_uniformly(pos_patches, dir_patches, samples_coarse)

                pixel_values_simulator, densities = simulator_approximation(samples, get_single_densities=True)
                pixel_values_simulator = pixel_values_simulator.reshape((num_patches, width, height))
                samples = self.data_loader.inverse_transform_sampling(densities.detach(), pos_patches, dir_patches, samples_coarse, samples_fine)

                pixel_values = simulator_model(samples)
                pixel_values = pixel_values.reshape((num_patches, width, height))

                img_prediction_small[x:x_end, y:y_end, 0] = pixel_values_simulator.detach().cpu().numpy()
                img_prediction[x:x_end, y:y_end, 0] = pixel_values.detach().cpu().numpy()
                img_labels[x:x_end, y:y_end] = label_patches.detach().cpu().numpy()
                y = y+bs
                if(break_y_loop and not break_x_loop):
                    y = 0
                    x = x+ bs
                    break_y_loop = False
                elif(break_x_loop and break_y_loop):
                    break
            
            if(break_x_loop): 
                break                
        
        self.save_img(self.log_folder_path+"Prediction_small.png", img_prediction_small)
        self.save_img(self.log_folder_path+"Prediction_large.png", img_prediction)
        self.save_img(self.log_folder_path+"Label.png", img_labels)

        return img_prediction, img_prediction_small, img_labels

    def validate_volumeslice(self, gt_slice, predicted_slice):
        gt_slice = gt_slice.astype(np.float32)
        predicted_slice = predicted_slice.astype(np.float32)

        psnr_val = psnr(gt_slice, predicted_slice)[0]
        mse_val = mse(gt_slice, predicted_slice)[0]
        dssim_val = dssim(gt_slice, predicted_slice)[0]

        return psnr_val, mse_val, dssim_val
