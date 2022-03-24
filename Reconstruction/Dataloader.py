import gzip, pickle
import numpy as np
import glob
import os

from Utils import *

import torch


np.random.seed(seed=42)
torch.manual_seed(42)

PARAMS = "/params/"

########################################
# Loads Training Data from pkl files   #
########################################

class Data_Loader():
    def __init__(self, path, num_validation, pos_enc, gt_path = None):
        self.path = path
        self.num_validation = num_validation
        self.pos_enc = pos_enc
        try:
            self.w, self.h, self.ray_length, self.center_distance, self.is_norm_coords = read_pickle(str(self.path)+str(PARAMS)+"_wh.pkl")
        except: 
            try:
                self.w, self.h, self.ray_length, self.center_distance = read_pickle(str(self.path)+str(PARAMS)+"_wh.pkl")
                self.is_norm_coords = False
            except: 
                self.w, self.h, self.ray_length = read_pickle(str(self.path)+str(PARAMS)+"_wh.pkl")
                self.center_distance = self.ray_length
                self.is_norm_coords = False
        self.gt_path = gt_path
    
    
    def combined_norm(self, gt_imgs, noisy_imgs):
        maximum = np.max([gt_imgs, noisy_imgs])
        minimum = np.min([gt_imgs, noisy_imgs])

        gt_imgs = (gt_imgs-minimum)/(maximum-minimum)
        noisy_imgs = (noisy_imgs-minimum)/(maximum-minimum)
        return gt_imgs, noisy_imgs

    def independent_norm(self, gt_imgs, noisy_imgs):
        gt_imgs = (gt_imgs-np.min(gt_imgs))/(np.max(gt_imgs)-np.min(gt_imgs))
        noisy_imgs = (noisy_imgs-np.min(noisy_imgs))/(np.max(noisy_imgs)-np.min(noisy_imgs))
        return gt_imgs, noisy_imgs

    def load_from_path(self, paths, is_validation = False):
        print("Image size: ("+str(self.w)+","+str(self.h)+")")
        positions = []
        directions = []
        labels = []
        length_arr = 0
        angles = np.zeros((len(paths), 1))

        for i,path in enumerate(paths): 
            file_name = os.path.basename(path)
            if(file_name == "_wh.pkl"):
                continue
            if(is_validation):
                img_positions, img_directions, img_labels = read_pickle(path)
            else: 
                try:
                    img_positions, img_directions, img_labels, angle = read_pickle(path)
                except: 
                    img_positions, img_directions, img_labels = read_pickle(path)
                    print(str(i)+": Did not load angle for Blur!")
                    angle = 0

            img_positions = img_positions.reshape(self.w,-1,3)
            img_directions = img_directions.reshape(self.w,-1,3)
            img_labels = img_labels.reshape(self.w,-1,1)
            length_arr += img_labels.shape[1]

            positions.append(img_positions)
            directions.append(img_directions)
            labels.append(img_labels)

            if(not is_validation):
                angles[i] = angle

        np_positions = np.zeros((self.h, length_arr, 3))
        np_directions = np.zeros((self.h, length_arr, 3))
        np_labels = np.zeros((self.h, length_arr, 1))
        img_width = []
        idx = 0
        for i in range(len(positions)):
            curr_length = positions[i].shape[1]
            img_width.append(curr_length)
            np_positions[:,idx:idx+curr_length,:] = positions[i]
            np_directions[:,idx:idx+curr_length,:] = directions[i]
            np_labels[:,idx:idx+curr_length,:] = labels[i]

            idx += curr_length
        if(not is_validation):
            return np_positions, np_directions, np_labels, angles, np.array(img_width)
        else:
            return np_positions, np_directions, np_labels


    #loads validation data and training data from self.path (using np.random)
    #also saves names of validation file(s)
    def load_data(self):
        if(self.gt_path != None):
            gt_paths = np.array(glob.glob(self.gt_path+"*.pkl"))
        
        paths = np.array(glob.glob(self.path+"*.pkl"))
        if(os.path.isdir(self.path+"/val_files/")): 
            val_files = np.array(glob.glob(self.path+"/val_files/*.pkl"))
            train_files = paths
        else: 
            #get validation files
            val_files_idx = np.random.randint(0, len(paths), size= self.num_validation)
            val_files = np.take(paths, val_files_idx)
            train_files = paths[~np.isin(paths,val_files)]
        print(val_files)

        v_img_width = self.w
        v_positions, v_directions, v_labels, v_angles, v_img_width = self.load_from_path(val_files)
        t_positions, t_directions, t_labels, t_angles, t_img_width = self.load_from_path(train_files)
        print("Shape training positions, shape validation positions:")
        print(t_positions.shape)
        print(v_positions.shape)

        if(self.gt_path != None):
            if(os.path.isdir(self.gt_path+"/val_files/")):
                val_files_gt = np.array(glob.glob(self.gt_path+"/val_files/*.pkl"))
            else: 
                val_files_gt = np.take(gt_paths, val_files_idx)
            _,_,val_gt,_,_ = self.load_from_path(val_files_gt)
            val_files = val_files_gt
        else: 
            val_gt = v_labels
        return t_positions, t_directions, t_labels, t_angles, t_img_width, v_positions, v_directions, v_labels, v_angles, v_img_width, val_files, val_gt
    
    #in: positions  [bs*ps, 3]
    #in: directions [bs*ps, 3]
    #out: samplepoints   [bs*ps, samples, pos_enc]
    def sample_uniformly(self, positions, directions, samples):
        bs = positions.shape[0]
        bin_size = self.ray_length/samples
        bins = torch.arange(start=0, end=self.ray_length, step=bin_size)
        bins = torch.tile(bins, (bs,1))

        uniform_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([bin_size]))
        uniform_dist = torch.squeeze(uniform_dist.rsample((bs,samples)))
        bins = bins+uniform_dist
        bins, _ = torch.sort(bins, dim = -1)
        bins = torch.tile(bins, (3,1,1))
        bins = bins.permute(1,2,0).cuda()

        directions = torch.tile(directions, (samples,1,1)).permute(1,0,2)
        positions = torch.tile(positions, (samples,1,1)).permute(1,0,2)
        
        out = positions+(directions*bins)

        out = pos_enc_fct(self.pos_enc//2, out)
        out = out.type(torch.float32)
        return out.cuda()

    #usually: samples fine = 2*samplescoarse
    #shape densities: bs,samples_coarse,1
    def inverse_transform_sampling(self, densities, positions, directions, samples_coarse, samples_fine):
        bin_size = self.ray_length/samples_coarse
        bs = positions.shape[0]

        discrete_ray_coarse = torch.arange(0, self.ray_length, step=bin_size) #shape [samples_coarse,1]
        densities[densities<0] = 0
        norm_densities = torch.sum(densities, 1)
        norm_densities = torch.tile(norm_densities.squeeze(), (densities.shape[1],1)).transpose(1,0)
        densities = torch.div(densities.squeeze(),norm_densities.squeeze())
        densities[torch.isnan(densities)] = 1/(samples_coarse)
        
        idx_fine = torch.multinomial(torch.squeeze(densities), samples_fine, replacement=True) #shape [bs,samples_fine,1]

        bins_coarse = discrete_ray_coarse[idx_fine] # shape [bs, samples_fine, 1]
        bins_fine = torch.tile(discrete_ray_coarse, (bs,1))
        bins = torch.cat((bins_coarse, bins_fine), dim = 1)

        samples = bins.shape[1]
        uniform_dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([bin_size]))
        uniform_dist = torch.squeeze(uniform_dist.rsample((bs,samples)))
        bins = bins+uniform_dist        
        bins, _ = torch.sort(bins, dim = -1)
        bins = torch.tile(bins, (3,1,1)).cuda()
        bins = bins.permute((1,2,0))

        directions = torch.tile(directions, (samples,1,1)).permute(1,0,2)
        positions = torch.tile(positions, (samples,1,1)).permute(1,0,2)
        
        out = positions+(directions*bins)
        out = pos_enc_fct(self.pos_enc, out)
        out = out.type(torch.float32)
        return out.cuda()
