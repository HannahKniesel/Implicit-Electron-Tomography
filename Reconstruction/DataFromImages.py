from PIL import Image
import glob
import numpy as np
import gzip, pickle, pickletools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os 
import datetime
import sys

PARAMS = "/params/"

########################################
# Prepares Training Data from Images   #
########################################

class Data_From_Images():
    def __init__(self, micrograph_path, save_to, slice, norm_coords, synthetic):
        self.micrograph_path = micrograph_path
        self.save_to = save_to
        self.slice = slice/2
        self.norm_coords = norm_coords
        self.synthetic = synthetic

        paths = glob.glob(self.micrograph_path+"/*.tif")
        if(len(paths)>0):
            img = np.array(Image.open(paths[0]))
            self.w = img.shape[0]
            self.h = img.shape[1]
        else:
            #remove spaces from filenames, when no filenames are found
            if(len(os.listdir(self.micrograph_path))>0):
                for f in os.listdir(self.micrograph_path):
                    r = f.replace(" ","_")
                    if(r != f):
                        os.rename(self.micrograph_path+"/"+f, self.micrograph_path+"/"+r)
                print("Changed filenames, so no spaces are contained.")

                #retry to load micropaths
                paths = glob.glob(self.micrograph_path+"/*.tif")
                if(len(paths)>0):
                    img = np.array(Image.open(paths[0]))
                    self.w = img.shape[0]
                    self.h = img.shape[1]
                else: 
                    print("ERROR:: Could not set width and height of micrographs")
                    print("Only found paths: "+str(paths))
                    print("Searched in file: "+str(self.micrograph_path+"/*.tif"))     
                    print(os.listdir(self.micrograph_path))           
                    sys.exit(-1)
        print()
        print("Loaded micropaths: ")
        print(str(paths)+"\n")
        return
    
    # converts a 3 channel rgb image to 1 channel grayscale image
    def rgb_to_gray(self, img):
        return img[:,:,0]*0.2126 + img[:,:,1]*0.7152 + img[:,:,2]*0.0722


    # loads all projections and normalizes to a range of 0-1
    def load_images(self):
        paths = glob.glob(self.micrograph_path+"/*.tif")

        #load all images in grayscale to np array
        images = np.zeros((len(paths), self.w, self.h))
        for i,path in enumerate(paths): 
            img = np.array(Image.open(path))
            if(len(img.shape)>=3):
                img = self.rgb_to_gray(img)
            images[i, :, :] = img

        #min max scaling, so all images are in range [0,1]
        images = (images-np.min(images))/(np.max(images)-np.min(images))
        return images

    #loads angles in degrees from rawtlt file
    def load_angles(self):
        path = glob.glob(self.micrograph_path+"/*rawtlt")

        try:
            # Open a file: file
            file = open(path[0] ,mode='r')
        except: 
            print("ERROR:: .rawtlt file was not found")
            sys.exit(-1)

        content = file.read()
        arr = np.array(content.split('\n'))
        arr = arr[arr != '']
        arr = arr.astype(float)

        # close the file
        file.close()
        return arr

    # rotates batch of points counterclockwise by angle 
    # [pos] ...batch of 3D points with shape [bs, 3]
    # [angle] ...angle in degree
    def rotate_y(self, pos, angle):
        angle = np.deg2rad(angle)
        oy, oz = 0, 0
        px, py, pz = pos.T
        qy = oy + np.cos(angle) * (py - oy) - np.sin(angle) * (pz - oz) 
        qz = oz + np.sin(angle) * (py - oy) + np.cos(angle) * (pz - oz)
        new_point = np.array([px, qy, qz])
        return new_point.T

    def compute_norm(self, radius):
        return np.sqrt(1+(radius**2))
    

    def save_data(self):        
        center_distance = np.sqrt((1**2)+(self.slice)**2)
        if(self.synthetic):
            center_distance = np.sqrt(2)        

        init_img = np.zeros((self.w,self.h,3))
        init_img[:,:,2] = center_distance #move center distance along z-axis

        stepsize = 2/(np.max([self.w, self.h]))

        x_coords = np.linspace(start = -1*stepsize*self.w/2, stop = (1*stepsize*self.w/2), num = self.w)
        y_coords = np.linspace(start = -1*stepsize*self.h/2, stop = (1*stepsize*self.h/2), num = self.h) 

        x_coords, y_coords = np.meshgrid(x_coords,y_coords)
        init_img[:,:,0] = x_coords.T
        init_img[:,:,1] = y_coords.T

        #flatten init_image to shape [bs, 3]
        init_img = init_img.reshape(-1,3)

        init_dir = np.array([0,0,-1])
        init_dir = np.expand_dims(init_dir, axis = 0) #add batch dimension

        angles = self.load_angles()
        images = self.load_images()

        if(len(angles) != images.shape[0]):
            print("ERROR:: Number angles ("+str(len(angles))+") does not match number images ("+str(images.shape[0])+")")
            sys.exit(-1)
        
        fig, ax = plt.subplots()
        plt.xlabel("y")
        plt.ylabel("z")
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        ax.set_aspect('equal', adjustable='box')

        for i, angle in enumerate(tqdm(angles)): 
            labels = images[i,:,:]
            
            img_positions = self.rotate_y(init_img, angle)
            img_positions = img_positions.reshape(self.w, self.h, 3)
            
            img_direction = self.rotate_y(init_dir, angle) #shape [1,3]
            img_direction = np.tile(np.squeeze(img_direction), (self.w, self.h, 1)).reshape(self.w, self.h, -1) #shape [w*h, 3]

            i_str = str(i)
            while(len(i_str)< len(str(len(angles)))):
                i_str = "0"+i_str

            if(not self.synthetic):
                bool_arr = (img_positions[:,:,2]>self.slice)
                bool_arr = np.tile(np.expand_dims(bool_arr, axis = -1), (1,1,3))
                img_positions = img_positions[bool_arr].reshape(self.w,-1, 3)
                labels = labels[bool_arr[:,:,0]].reshape(self.w,-1)
                img_direction = img_direction[bool_arr].reshape(self.w,-1, 3)       
            distance = 2*center_distance           

            end_positions = img_positions+distance*img_direction
            if(not self.synthetic):
                bool_arr = (end_positions[:,:,2]<-1*self.slice) 
                bool_arr = np.tile(np.expand_dims(bool_arr, axis = -1), (1,1,3))
                img_positions = img_positions[bool_arr].reshape(self.w,-1, 3)
                end_positions = end_positions[bool_arr].reshape(self.w,-1, 3)
                labels = labels[bool_arr[:,:,0]].reshape(self.w,-1)
                img_direction = img_direction[bool_arr].reshape(self.w,-1, 3)
           
            if(img_positions.shape[1]>0):
                mid_h = img_positions.shape[0]//2
                mid_w = img_positions.shape[1]//2
                if(self.norm_coords):
                    norm_rad = self.compute_norm(center_distance)
                    img_positions[:,:,1:] = (img_positions[:,:,1:]/norm_rad)
                    distance = distance/norm_rad
                else: 
                    norm_rad = 1

                self.save_as_pickle([img_positions, img_direction, labels, angle], self.save_to+"/"+i_str+"_angle_"+str(int(angle)))
                plt.scatter(img_positions[:,:,1], img_positions[:,:,2])
                plt.plot([img_positions[mid_h, mid_w,1], img_positions[mid_h, mid_w,1] + img_direction[0,0,1]*distance], [img_positions[mid_h, mid_w,2], img_positions[mid_h, mid_w,2] + img_direction[0,0,2]*distance])
        rect0 = patches.Rectangle((-1/norm_rad,-1/norm_rad), 2/norm_rad, 2/norm_rad, linewidth=1, edgecolor='g', facecolor='none', zorder=10, label="Reconstruction Space")
        rect1 = patches.Rectangle((-1,-1), 2, 2, linewidth=1, edgecolor='b', facecolor='none', zorder=2, label="Range [-1,1]")
        rect2 = patches.Rectangle((-1,-self.slice/norm_rad), 2, 2*self.slice/norm_rad, linewidth=1, edgecolor='r', facecolor='r', zorder=2, alpha=0.5, label="Slice Thickness")

        ax.add_patch(rect0)
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        plt.legend()
        plt.savefig(self.save_to+"/rays.png")
        self.save_as_pickle([self.w, self.h, distance, center_distance, self.norm_coords], self.save_to+PARAMS+"/_wh")
        print("Saved files to: "+str(self.save_to))
        return True

    
    #saves list of values into pkl file
    def save_as_pickle(self, lst, path):
        with gzip.open(str(path+".pkl"), 'wb') as f:
            pickled = pickle.dumps(lst)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
            #pickle.dump(lst, f)
        if(type(lst) is list):
            lst.clear()
        return True

print("******************************")
print("Generate data from Micrographs")
print("******************************")

# Args Parser
parser = argparse.ArgumentParser(description='Generate positional data and labels')
parser.add_argument('--micrograph_path', type = str, default='./Micrographs/', help='Directory of micrographs and rawtlt (default: ./Micrographs/)')
parser.add_argument('--save_to', type = str, default='./TrainingData/', help='Directory to save data (default: ./TrainingData/)')
parser.add_argument('--slice', type = float, default=1, help='normalized slice thickness (default: 1)')
parser.add_argument("--norm_coords", default=False, action="store_true", help="Weather to normalize labels to range [-1,1]")
parser.add_argument("--synthetic", default=False, action="store_true", help="For synthetic data no culling is applied")


args = parser.parse_args()

print("Parameters:")
print(args)

# log folder
now = datetime.datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
log_folder_path = args.save_to+"/logs_"+date_time+"/"
if not os.path.isdir(log_folder_path):
    os.makedirs(log_folder_path)
    try:
        os.mkdir(args.save_to+PARAMS)
    except:
        pass

# Save args params
f = open(log_folder_path+"/args.txt", "w")
f.write(str(vars(args)))
f.close()

dfi = Data_From_Images(args.micrograph_path, args.save_to, args.slice, args.norm_coords, args.synthetic)
dfi.save_data()

#python DataFromImages.