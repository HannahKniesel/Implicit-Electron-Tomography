import numpy as np
import matplotlib.pyplot as plt
from random import sample 
from tqdm import tqdm
from scipy import ndimage
from transformations import rotation_matrix
import os
from scipy import ndimage
import argparse
import datetime

from Utils import *

def rotate_3D(vol):
    angle = np.deg2rad(np.random.randint(0,180))
    dim = vol.shape[0]
    ax = np.arange(dim)
    coords = np.meshgrid(ax,ax,ax)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([coords[0].reshape(-1)-float(dim)/2,     # x coordinate, centered
            coords[1].reshape(-1)-float(dim)/2,     # y coordinate, centered
            coords[2].reshape(-1)-float(dim)/2,     # z coordinate, centered
            np.ones((dim,dim,dim)).reshape(-1)])    # 1 for homogeneous coordinates

    mat=rotation_matrix(angle,(0,1,0))
    # apply transformation
    transformed_xyz=np.dot(mat, xyz)

    # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
    x=transformed_xyz[0,:]+float(dim)/2
    y=transformed_xyz[1,:]+float(dim)/2
    z=transformed_xyz[2,:]+float(dim)/2

    x=x.reshape((dim,dim,dim))
    y=y.reshape((dim,dim,dim))
    z=z.reshape((dim,dim,dim))

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz=[y,x,z]

    # sample
    new_vol=ndimage.map_coordinates(vol,new_xyz, order=0)
    return new_vol

class Virus():
    def __init__(self, path, vol_size, iterations, max_num_virus, slice_thickness, volume):
        self.path = path 
        self.virus = open_mrc(self.path)
        self.virus = min_max(self.virus)
        self.virus = ndimage.zoom(self.virus, 0.2)
        print(self.virus.shape)
        self.vol_size = vol_size
        if(volume is None):
            self.volume = np.zeros((vol_size, vol_size, vol_size),dtype=np.float32)
        else: 
            self.volume = volume
        self.iterations = iterations
        self.max_num_virus = max_num_virus
        self.slice_thickness = slice_thickness
        self.place_virus()

    def place_virus(self):
        num_elems = 0
        v_size = self.virus.shape[0]

        start_slice = (self.vol_size-self.slice_thickness)//2
        end_slice = start_slice + self.slice_thickness
        for i in tqdm(range(self.iterations)):
            pos_x = np.random.randint(low = start_slice, high = end_slice-v_size, size = (1,))
            pos_y = np.random.randint(low = 0, high = self.vol_size-v_size, size = (1,))
            pos_z = np.random.randint(low = 0, high = self.vol_size-v_size, size = (1,))

            pos = np.squeeze(np.stack([pos_x, pos_y, pos_z]))
            patch = self.volume[pos[0]:pos[0]+v_size, pos[1]:pos[1]+v_size, pos[2]:pos[2]+v_size]
            if(np.unique(patch).shape[0]>1):#np.sum(patch)>0):
                continue
            else: 
                num_elems += 1
                virus = rotate_3D(self.virus)
                self.volume[pos[0]:pos[0]+v_size, pos[1]:pos[1]+v_size, pos[2]:pos[2]+v_size] += virus
                if(num_elems >= self.max_num_virus):
                    break
        print("Number virus set: "+str(num_elems))
        return 


class Membrane_Elipsis():
    def __init__(self, r, patch=None):
        self.thickness = np.random.uniform(low=0.85, high=0.95)
        self.thickness = self.thickness

        density_membrane =  np.random.uniform(low=0.5, high=0.98) # high density 
        density_fluids = np.random.uniform(low=0.1, high=0.5)
        #inside 0.5 
        which_r = sample([0,1,2],1)
        rs = np.random.uniform(low = 0.7*r, high = r, size=(3,))
        rs[which_r] = r
        r_1,r_2,r_3 = rs

        volume = np.zeros((r*2, r*2, r*2), dtype=np.float32)

        x = np.arange(-r,r)
        y = np.arange(-r,r)
        z = np.arange(-r,r)
        x,y,z = np.meshgrid(x,y,z)

        inner_bool = (x**2)/(r_1*self.thickness)**2 + (y**2)/(r_2*self.thickness)**2 + (z**2)/(r_3*self.thickness)**2 < 1
        outer_bool = (x**2)/(r_1**2) + (y**2)/(r_2**2) + + (z**2)/(r_3)**2 < 1
        volume[outer_bool] = density_membrane
        volume[inner_bool] = density_fluids

        if(not(patch is None)):
            if(np.unique(patch[inner_bool]).shape[0]>1):
                self.volume = None
                return
        self.volume = volume



class Ellipses_Volume():
    def __init__(self, size, slice_thickness, iterations, volume):

        self.large_iters = iterations//8
        self.small_iters = iterations - self.large_iters
        self.start_slice = (size-slice_thickness)//2
        self.end_slice = self.start_slice + slice_thickness

        num_items = 0
        self.size = size
        self.slice_thickness = slice_thickness
        if(volume is None):
            self.volume = np.zeros((self.size, self.size, self.size), dtype=np.float32)
        else:
            self.volume = volume       
        return

    def place_large(self):
        max_elem_size = int((0.8*self.slice_thickness-1)//2)
        min_elem_size = int((0.3*self.slice_thickness)//2)
        rs = np.random.randint(low=min_elem_size, high=max_elem_size, size=(iterations//4))
        rs = np.sort(rs)[::-1]
        num_items = 0
        for i in tqdm(range(self.large_iters), desc="Place large ellipses"):
            r = rs[i]
            size = 2*r
            pos_x = np.random.randint(low = self.start_slice, high = self.end_slice-size, size = (1,))
            pos_y = np.random.randint(low = 0, high = self.size-size, size = (1,))
            pos_z = np.random.randint(low = 0, high = self.size-size, size = (1,))

            if(pos_x+size > self.end_slice):
                continue
            if(pos_x < self.start_slice):
                continue

            pos = np.squeeze(np.stack([pos_x, pos_y, pos_z]))
            patch = self.volume[pos[0]:pos[0]+size, pos[1]:pos[1]+size, pos[2]:pos[2]+size]
            obj = Membrane_Elipsis(r=r, patch=patch)
            if(not(obj.volume is None)):
                vol = rotate_3D(obj.volume)
                self.volume[pos[0]:pos[0]+size, pos[1]:pos[1]+size, pos[2]:pos[2]+size] += vol
                num_items += 1
        
        print("Number of large ellipses placed: "+str(num_items))
        return self.volume

    def place_small(self):
        num_items = 0
        max_elem_size = int((0.5*slice_thickness)//2)
        min_elem_size = int((0.1*slice_thickness)//2)
        rs = np.random.randint(low=min_elem_size, high=max_elem_size, size=(iterations))
        rs = np.sort(rs)[::-1]
        for i in tqdm(range(self.small_iters), desc="Place small ellipses"):
            r = rs[i]
            size = 2*r
            pos_x = np.random.randint(low = self.start_slice, high = self.end_slice-size, size = (1,))
            pos_y = np.random.randint(low = 0, high = self.size-size, size = (1,))
            pos_z = np.random.randint(low = 0, high = self.size-size, size = (1,))

            if(pos_x+size > self.end_slice):
                continue
            if(pos_x < self.start_slice):
                continue

            pos = np.squeeze(np.stack([pos_x, pos_y, pos_z]))
            patch = self.volume[pos[0]:pos[0]+size, pos[1]:pos[1]+size, pos[2]:pos[2]+size]
            obj = Membrane_Elipsis(r=r, patch=patch)
            if(not(obj.volume is None)):
                vol = rotate_3D(obj.volume)
                self.volume[pos[0]:pos[0]+size, pos[1]:pos[1]+size, pos[2]:pos[2]+size] += vol
                num_items += 1
        
        print("Number of small ellipses placed: "+str(num_items))
        return self.volume





class MakeCell():
    def __init__(self, virus_path, vol_size, iterations, max_num_virus, slice_thickness):
        self.volume = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)
        self.slice_thickness = slice_thickness
        self.size = vol_size
        
        ellipses = Ellipses_Volume(vol_size, slice_thickness, iterations, self.volume)
        self.volume = ellipses.place_large()
        # self.volume = ellipses.volume

        virus = Virus(virus_path, vol_size, iterations, max_num_virus, slice_thickness, self.volume)
        self.volume = virus.volume

        self.volume = ellipses.place_small()

        self.apply_slice()

    def apply_slice(self):
        start_slice = (self.size-self.slice_thickness)//2
        end_slice = start_slice + self.slice_thickness
        self.volume[:start_slice,:,:] = 0
        self.volume[start_slice:end_slice, :, :] += 0.2
        self.volume[end_slice:, :,:] = 0
        return

if __name__ == "__main__":
    print("******************************")
    print("Generate Phantom Volume")
    print("******************************")
    parser = argparse.ArgumentParser(description='Train the reconstruction network')

    #Starting iterations
    parser.add_argument('--resolution', type = int, default=512, help='Resolution of phantom volume (default: 512)')
    parser.add_argument('--slice_thickness', type = int, default=300, help='Sample thickness in voxels (default: 300)')
    parser.add_argument('--max_num_virus', type = int, default=200, help='maximum number of virus placed (default: 200)')
    parser.add_argument('--virus_path', type = str, default="./VirusPDB/6mid_15A.mrc", help='Path where to load virus density map from (default: ./VirusPDB/6mid_15A.mrc)')
    parser.add_argument('--iterations', type = int, default=1000, help='Maximum iterations to try place objects. (default: 1000)')
    parser.add_argument('--out', type = str, default='./PhantomVolume/', help='Output directory')


    args = parser.parse_args()

    print("Parameters:")
    print(args)

    # log folder
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    log_folder_path = args.out+"/logs_"+date_time+"/"
    if not os.path.isdir(log_folder_path):
        os.makedirs(log_folder_path)

    # Save args params
    f = open(log_folder_path+"/args.txt", "w")
    f.write(str(vars(args)))
    f.close()

    path =  args.virus_path 
    vol_size = args.resolution 
    iterations = args.iterations
    slice_thickness = args.slice_thickness
    max_num_virus = args.max_num_virus

    cell = MakeCell(path, vol_size, iterations, max_num_virus, slice_thickness)

    volume = cell.volume.transpose((1,0,2))

    base_path = log_folder_path
    slices_path = "/Slices/"
    raw_path = ""

    os.makedirs(base_path+slices_path, exist_ok=True)

    save_raw(base_path+raw_path, volume)
    save_tif_stack(volume, base_path+slices_path)
