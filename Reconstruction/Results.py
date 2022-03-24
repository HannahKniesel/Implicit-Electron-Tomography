import argparse
import datetime
import os
import glob
from PIL import Image
import numpy as np
import mrcfile
import torch
from tqdm import tqdm 
import skimage.metrics as m
import pickle
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R

from scipy.ndimage import zoom

from Models.EMSimulator import EM_Simulator
from Utils import *

np.random.seed(42)
torch.manual_seed(42)

def load_model(training_state_path):
    checkpoint = torch.load(training_state_path)
    pos_enc = checkpoint['pos_enc']
    model = EM_Simulator(pos_enc, large=True).cuda() 
    model.load_state_dict(checkpoint['model_large_state_dict'])
    model.eval()
    return model, pos_enc

def save_dict(dic, path):
    dic_txt = dic.copy()
    a_file = open(path+"/data.pkl", "wb")
    pickle.dump(dic, a_file)
    a_file.close()

    with open(path+'/data.txt', 'w') as file:
        file.write(json.dumps(dic_txt))

def psnr(vol_gt, vol, slices=True):
    if(slices):
        mpsnr = []
        for proj_gt, proj in zip(vol_gt,vol):
            mpsnr.append(m.peak_signal_noise_ratio(proj_gt,proj))
        std_psnr = np.std(mpsnr)
        mpsnr = np.mean(mpsnr)
    else:
        mpsnr = m.peak_signal_noise_ratio(vol_gt, vol)
        std_psnr = 0
    return mpsnr, std_psnr

def mse(vol_gt, vol, slices=True):
    if(slices):
        mmse = []
        for proj_gt, proj in zip(vol_gt,vol):
            mmse.append(m.mean_squared_error(proj_gt,proj))
        std_mse = np.std(mmse)
        mmse = np.mean(mmse)
    else:
        mmse = m.mean_squared_error(vol_gt, vol)
        std_mse = 0
    return mmse, std_mse

#if slices True compute error distribution over fist dimension of volume
def ssim_dssim(vol_gt, vol, slices=True):
    if(slices):
        mssim = []
        mdssim = []
        for proj_gt, proj in zip(vol_gt, vol):
            ssim = m.structural_similarity(proj_gt, proj, multichannel=False)
            mssim.append(ssim)
            mdssim.append((1-ssim)/2)
        std_ssim = np.std(mssim)
        std_dssim = np.std(mdssim)
        mssim = np.mean(mssim)
        mdssim = np.mean(mdssim)
    else: 
        mssim = m.structural_similarity(vol_gt, vol, multichannel=False)
        mdssim = (1-mssim)/2
        std_ssim = 0
        std_dssim = 0
    
    return mssim, std_ssim, mdssim, std_dssim


class Results():
    def __init__(self, gt_path, state_path, format, log_path, num_projections, resolution, grid):
        self.resolution = resolution
        #load gt volume
        gt_file = gt_path+"/*"+format
        if(format ==".mrc"):
            self.gt = open_mrc(glob.glob(gt_file))
        elif(format ==".raw"):
            self.gt = open_raw(glob.glob(gt_file))
        self.gt = min_max(self.gt).astype(np.float32) 
        print("GT Volume loaded. Max: "+str(np.max(self.gt))+", Min: "+str(np.min(self.gt))+" | Resolution: "+str(self.gt.shape))

        if(self.resolution>0):
            zoom_level = self.resolution/self.gt.shape[0]
            self.gt = zoom(self.gt, (zoom_level, zoom_level, zoom_level))
            self.gt[self.gt>1] = 1
            self.gt[self.gt<0] = 0
            print("Rescaled to resolution: "+str(self.gt.shape))
        
        #load model
        if(grid):
            self.model = load_explicit_model(state_path)
        else:
            self.model, self.pos_enc = load_implicit_model(state_path)

        #volume eval
        v = Volume(self.gt, self.model, log_path, self.pos_enc, grid, state_path)
        print("Finished Volume Evaluation")
        torch.cuda.empty_cache()
        
        #projection eval
        p = ProjectionMaker(self.gt, v.volume_model, self.pos_enc, num_projections, projection_shape = None, log_path = log_path)

        print("Log path: "+str(log_path))


class Volume():
    def __init__(self, volume_gt, model, log_path, pos_enc, grid, model_path):
        self.pos_enc = pos_enc
        self.volume_gt = volume_gt
        self.model = model
        self.resolution = volume_gt.shape[0]
        self.model_path = model_path
        if(grid):
            self.volume_model = self.make_volume_explicit()
        else:
            self.volume_model = self.make_volume_implicit()
        dic = self.evaluate()
        print()
        print("Volume: "+str(dic))
        print()

        volume_path = log_path +"/Volume/"
        os.mkdir(volume_path)
        save_dict(dic, volume_path)
        save_raw(volume_path, self.volume_model)
        print("Done. \n")

        fig,axs = plt.subplots(3,2)
        axs[0,0].set_title("xz")
        axs[0,0].imshow(self.volume_model[:,self.volume_model.shape[0]//2,:])
        axs[0,1].set_title("xz")
        axs[0,1].imshow(self.volume_gt[:,self.volume_gt.shape[0]//2,:])
        axs[1,0].set_title("yz")
        axs[1,0].imshow(self.volume_model[self.volume_model.shape[0]//2,:,:])
        axs[1,1].set_title("yz")
        axs[1,1].imshow(self.volume_gt[self.volume_gt.shape[0]//2,:,:])
        axs[2,0].set_title("xy")
        axs[2,0].imshow(self.volume_model[:,:,self.volume_model.shape[0]//2])
        axs[2,1].set_title("xy")
        axs[2,1].imshow(self.volume_gt[:,:,self.volume_gt.shape[0]//2])

        plt.savefig(volume_path+"/Slices.png")
        plt.close()


    def make_volume_explicit(self):
        scale = recon.shape[0]/self.resolution
        recon = zoom(recon, scale)
        return min_max(recon).astype(np.float32)

    def make_volume_implicit(self):
        coordinates = make_coordinates(self.resolution)
        try:
            is_norm_coords, center_distance = load_norm_coords_params(self.model_path)
        except: 
            is_norm_coords = False
        print("Normalize coordinates: "+str(is_norm_coords))
        
        if(is_norm_coords):
            r = center_distance
            norm = np.sqrt(1+(r**2))
            coordinates[:,:,:,1:] = (coordinates[:,:,:,1:]/norm)
        print("Coordinate resolution: "+str(coordinates.shape))

        coordinates = coordinates.reshape(-1,coordinates.shape[-1])
        bs = 1024
        idx = 0
        vol_large = np.zeros((coordinates.shape[0], 1),dtype=np.float32)
        pbar = tqdm(total=coordinates.shape[0], desc="Predict Volume")
        while(idx+bs < coordinates.shape[0]):
            batch = coordinates[idx:idx+bs,:].cuda()
            batch = self.pos_enc_fct(self.pos_enc, batch)
            #large network
            prediction = self.model(batch, is_training = False)
            vol_large[idx:idx+bs,:] = prediction.cpu().detach().numpy().astype(np.float32)
            idx = idx+bs
            pbar.update(bs)
        vol_large = vol_large.reshape(self.resolution, self.resolution, self.resolution)
        vol_large[vol_large>1] = 1
        vol_large[vol_large<0] = 0      
        return min_max(vol_large).astype(np.float32)
        
    def evaluate(self):
        print("Evaluate MSE, PSNR, SSIM, DSSIM")
        mmse,smse = mse(self.volume_gt, self.volume_model, slices = False)
        mpsnr,spsnr = psnr(self.volume_gt, self.volume_model, slices = False)
        mssim, sssim, mdssim, sdssim = ssim_dssim(self.volume_gt, self.volume_model, slices = False)
        eval_dict = {
            "m_ssim": mssim,
            "s_ssim": sssim,
            "m_dssim": mdssim,
            "s_dssim": sdssim,
            "m_psnr": mpsnr,
            "s_psnr": spsnr, 
            "m_mse": mmse,
            "s_mse": smse
        }
        return eval_dict

    def pos_enc_fct(self, num_pos_enc, x):
        rets = [x]
        for i in range(num_pos_enc):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn((np.pi*2.**i)*x))
        return torch.cat(rets,-1)


class ProjectionMaker():
    def __init__(self, volume, model, pos_enc, num_projections, projection_shape = None, log_path = None):
        self.volume = volume
        self.model = model
        self.projection_shape = projection_shape

        self.num_projections = num_projections

        self.pos_enc = pos_enc

        if(self.projection_shape):
            self.w,self.h = self.projection_shape
        else:
            self.w,self.h,_ = self.volume.shape
        
        self.rot = np.zeros((num_projections,3))
        self.rot[:,:2] = np.random.randint(0,360,(num_projections,2))

        projections_path = log_path+"/Projections/"
        os.mkdir(projections_path)
        print("Compute projections...")
        self.projections_gt, self.projections_model = self.projections(volume = self.volume, model = self.model)
        dic = self.evaluate()
        print()
        print("Projections: "+str(dic))
        print()

        save_dict(dic, projections_path)

        gt_path = projections_path+"/GT/"
        predict_path = projections_path+"/Predict/"
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(predict_path, exist_ok=True)

        self.save_projections(gt_path, self.projections_gt)
        self.save_projections(predict_path, self.projections_model)
        print("Done. \n")

    def projections(self, volume, model):
        samples = 1024 
        r_length = (np.sqrt(2))
        r_length = r_length*self.w
        step_size = r_length/ samples

        init_img = self.init_view()

        gt_projections = []
        predicted_projections = []
        for j in range(self.num_projections):
            proj, dir = self.get_dir_origin(init_img, j)
            projection_volume = np.zeros((self.w,self.h))
            projection_model = np.zeros((self.w,self.h))

            #volume
            proj_volume = self.scale_to_vol(proj)
            for i in tqdm(range(samples)):
                sample = i*dir*step_size + proj_volume
                s = sample.astype(int)
                bool_arr = (s >= volume.shape[0]) | (s < 0)
                bool_arr = np.sum(bool_arr, axis = -1)
                cliped_s = np.clip(s, 0, volume.shape[0]-1)
                densities = volume[cliped_s[:,:,1], cliped_s[:,:,0], cliped_s[:,:,2]]
                projection_volume += np.where(bool_arr, 0, densities)*(step_size/100)

                densities = model[cliped_s[:,:,1], cliped_s[:,:,0], cliped_s[:,:,2]]
                projection_model += np.where(bool_arr, 0, densities)*(step_size/100)

            projection_volume = min_max(np.exp(-projection_volume))
            projection_model = min_max(np.exp(-projection_model))

            gt_projections.append(projection_volume)
            predicted_projections.append(projection_model)

        return gt_projections, predicted_projections


    def evaluate(self):
        print("Evaluate MSE, PSNR, SSIM, DSSIM")
        mmse,smse = mse(self.projections_gt, self.projections_model, slices = True)
        mpsnr,spsnr = psnr(self.projections_gt, self.projections_model, slices = True)
        mssim, sssim, mdssim, sdssim = ssim_dssim(self.projections_gt, self.projections_model, slices = True)
        eval_dict = {
            "m_ssim": mssim,
            "s_ssim": sssim,
            "m_dssim": mdssim,
            "s_dssim": sdssim,
            "m_psnr": mpsnr,
            "s_psnr": spsnr, 
            "m_mse": mmse,
            "s_mse": smse
        }
        return eval_dict

    def save_projections(self, path, projections):
        projections = min_max(projections)
        for i, proj in tqdm(enumerate(projections), desc="Save projections"):
            i_str = str(i)
            while(len(i_str)< len(str(projections.shape[0]))):
                i_str = "0"+i_str
            img = Image.fromarray((proj*255).astype(np.uint8))
            img.save(path+"/"+i_str+".png")
        return

    def get_dir_origin(self, init_proj, idx):
        proj = init_proj.reshape(-1,3)
        dir = np.array([0,0,-1])
        angle = self.rot[idx,:]
        x_rot = R.from_euler('x', angle[0], degrees=True)
        proj = x_rot.apply(proj)
        dir = x_rot.apply(dir)

        y_rot = R.from_euler('y', angle[1], degrees=True)
        proj = y_rot.apply(proj)
        dir = y_rot.apply(dir)

        proj = proj.reshape(self.w,self.h,3)
        return proj, (dir/(np.linalg.norm(dir)))

    def init_view(self):
        center_distance = np.sqrt(2)*self.w
        init_img = np.zeros((self.w,self.h,3))
        x_coords = np.linspace(start = -1, stop = 1, num =  self.w) * self.w
        y_coords = np.linspace(start = -1, stop = 1, num =  self.w) * self.w

        x_coords, y_coords = np.meshgrid(x_coords,y_coords)
        init_img[:,:,0] = x_coords.T
        init_img[:,:,1] = y_coords.T
        init_img[:,:,2] = center_distance
        return init_img

    def scale_to_vol(self, pos):
        return (pos+self.w)/2

    def scale_to_model(self, pos):
        return (pos/self.w)

    def pos_enc_fct(self, num_pos_enc, x):
        rets = [x]
        for i in range(num_pos_enc):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn((np.pi*2.**i)*x))
        return torch.cat(rets,-1)

    def predict(self, model, data):
        init_shape = data.shape
        data = data.reshape(-1,data.shape[-1])
        out = np.zeros((data.shape[0],1))
        bs = 2048
        start = 0
        end = start+bs
        while(start < data.shape[0]):
            if(end < data.shape[0]):
                input = data[start:end, :]
            else:
                input = data[start:, :]
            input = self.pos_enc_fct(self.pos_enc, torch.from_numpy(input.astype(np.float32))).cuda()
            out[start:end] = model(input, is_training=False).detach().cpu().numpy()

            start = end
            end = start+bs
        out = out.reshape(init_shape[0], init_shape[1])
        return out   

if __name__ == "__main__":

    print("******************************")
    print("Evaluate results")
    print("******************************")


    # Args Parser
    parser = argparse.ArgumentParser(description='Generate positional data and labels')
    parser.add_argument('--state_path', type = str, default="./TrainingRuns/", help='Path to file where to load the model from (default: "./TrainingRuns/")')
    parser.add_argument('--save_to', type = str, default="./Results/", help='Path where to save the evaluation to (default: "./Results/")')
    parser.add_argument('--volume_path', type = str, default='./Volumes/Fibril/', help='Path of phantom volume (default: ./Volumes/)')
    parser.add_argument('--format', type = str, default=".raw", help='Format of GT Volume. One of [.raw, .mrc] (default: .raw)')
    parser.add_argument('--name', type = str, default="_", help='Name of Training Run (default: "_")')
    parser.add_argument('--num_projections', type = int, default=20, help='Number projections (default: 100)')
    parser.add_argument('--resolution', type = int, default=-1, help='Resolution to use (default: resolution of GT volume)')
    parser.add_argument('--grid', action="store_true", default=False, help='Reconstruct based on explicit model (default: False)')

    args = parser.parse_args()

    print("Parameters:")
    print(args)


    # log folder
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    log_folder_path = args.save_to+args.name+"/"+date_time
    if not os.path.isdir(log_folder_path):
        os.makedirs(log_folder_path)
        

    # Save args params
    f = open(log_folder_path+"/args.txt", "w")
    f.write(str(vars(args)))
    f.close()

    r = Results(args.volume_path, args.state_path, args.format, log_folder_path, args.num_projections, args.resolution, args.grid)

    