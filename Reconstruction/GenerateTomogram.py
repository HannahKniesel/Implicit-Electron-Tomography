from scipy import ndimage
import numpy as np
from tqdm import tqdm
import datetime
import os
import argparse
import logging


from Utils import *

###############################################
# Generate Tomogram from trained models       #
###############################################

class Tomogram():
    def __init__(self, resolution, slices, training_state_path, log_folder_path, grid):
        self.resolution = resolution
        self.training_state_path = training_state_path
        self.save_to = log_folder_path
        self.slices = slices
        self.grid = grid

        logging.basicConfig(filename=log_folder_path+'/log.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

    def make_recon(self, model_path, resolution, slices):
        model, pos_enc = load_implicit_model(model_path)
        coordinates = make_coordinates(resolution, slices)
        try:
            is_norm_coords, center_distance = load_norm_coords_params(model_path)
        except: 
            print("Could not load data path")
            is_norm_coords = False
        print("Normalize coordinates: "+str(is_norm_coords))
        
        if(is_norm_coords):
            r = center_distance
            norm = np.sqrt(1+(r**2))
            coordinates[:,:,:,1:] = (coordinates[:,:,:,1:]/norm)
        print("Coordinate resolution: "+str(coordinates.shape))

        coordinates = coordinates.reshape(-1,coordinates.shape[-1])

        bs = 2048
        idx = 0
        vol_large = np.zeros((coordinates.shape[0], 1))

        pbar = tqdm(total=coordinates.shape[0], desc="Predict Volume")
        while(idx+bs <= coordinates.shape[0]):
            batch = coordinates[idx:idx+bs,:].cuda()
            batch = pos_enc_fct(pos_enc, batch)        
            prediction = model(batch, is_training = False)
            vol_large[idx:idx+bs,:] = prediction.cpu().detach().numpy()

            idx = idx+bs
            pbar.update(bs)

        if(idx <= coordinates.shape[0]):
            batch = coordinates[idx:,:].cuda()
            batch = pos_enc_fct(pos_enc, batch)
            prediction = model(batch, is_training = False)
            vol_large[idx:,:] = prediction.cpu().detach().numpy()

        vol_large = vol_large.reshape(resolution, resolution, slices, 1)

        vol_large[vol_large>1] = 1
        vol_large[vol_large<0] = 0
        return min_max(vol_large)


    def make_grid_recon(self, model_path, resolution, slices):
        recon = load_explicit_model(model_path)
        scale = recon.shape[0]/resolution
        recon = ndimage.zoom(recon, scale)
        if(slices > 0):
            start = (resolution//2) - (slices//2)
            end = start + slices
            recon = recon[:,:,start:end]
        return min_max(recon)

    def make(self):
        bits = (2**16)
        if(not self.grid):
            volume = self.make_recon(self.training_state_path, self.resolution, self.slices)
        else: 
            volume = self.make_grid_recon(self.training_state_path, self.resolution, self.slices)
        volume = (volume - np.min(volume))/(np.max(volume)- np.min(volume))
        
        #save tif sequence
        os.makedirs(self.save_to+"/TIF/", exist_ok=True)
        print("Saving TIF Sequence ...")
        save_tif_stack(volume.squeeze().transpose(2,0,1), self.save_to+"/TIF/")
        print("TIF Sequence was saved to: "+str(self.save_to+"/TIF/"))

        volume = 1 - volume
        volume = (volume*bits)-(2**15)

        print("Save raw file ...")
        data = min_max(volume)
        save_raw(self.save_to, data) 



if __name__ == "__main__":

    print("******************************")
    print("Make Volume")
    print("******************************")


    # Args Parser
    parser = argparse.ArgumentParser(description='Generate positional data and labels')
    parser.add_argument('--resolution', type = int, default=100, help='Resolution of the Volume (default: 100)')
    parser.add_argument('--state_path', type = str, default="./TrainingRuns/training_state.pth", help='Path to file where to load the model from (default: "./TrainingRuns/training_state.pth")')
    parser.add_argument('--save_to', type = str, default="./Tomograms/", help='Path where to save the tomogram to (default: "./Tomograms/")')
    parser.add_argument('--name', type = str, default="L2Noisy", help='Name of Training Run (default: "L2Noisy")')
    parser.add_argument('--slices', type = int, default=-1, help='Number of slices to reconstruction in z-dimension (default: -1)')
    parser.add_argument('--grid', action="store_true", default=False, help='Explicit model (default: False)')

    args = parser.parse_args()

    print("Parameters:")
    print(args)


    # log folder
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    log_folder_path = args.save_to+args.name
    if not os.path.isdir(log_folder_path):
        os.makedirs(log_folder_path)
        
    # Save args params
    f = open(log_folder_path+"/args.txt", "w")
    f.write(str(vars(args)))
    f.close()

    t = Tomogram(args.resolution, args.slices, args.state_path, log_folder_path, args.grid)
    t.make()
