import numpy as np
import torch
from tqdm import tqdm
import gzip, pickle
from PIL import Image
import skimage.metrics as m
import mrcfile
import glob


from Models.EMSimulator import EM_Simulator

# converts a 3 channel rgb image to 1 channel grayscale image
def rgb_to_gray(img):
    return img[:,:,0]*0.2126 + img[:,:,1]*0.7152 + img[:,:,2]*0.0722

def make_coords_central_slice(resolution):
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)

    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    coordinates = np.stack([X,Y,Z], axis=-1)
    coordinates = torch.Tensor(coordinates)#.cuda()
    
    return coordinates

def make_coordinates(resolution, slices = -1):
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)

    X,Y,Z = np.meshgrid(x,y,z)
    coordinates = np.stack([X,Y,Z], axis=-1)
    coordinates = torch.Tensor(coordinates)#.cuda()

    if(slices > 0):
        start = (resolution//2) - (slices//2)
        end = start + slices
        coordinates = coordinates[:,:,start:end, :]
    
    return coordinates

def pos_enc_fct(num_pos_enc, x):
    rets = [x]
    for i in range(num_pos_enc):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((np.pi*2.**i)*x))
    return torch.cat(rets,-1)

def load_norm_coords_params(path):
    checkpoint = torch.load(path)
    try:
        is_norm_coords = checkpoint['is_norm_coords']
        center_distance = checkpoint['center_distance']
    except: 
        is_norm_coords = False
        center_distance = -1
    return is_norm_coords, center_distance

def load_implicit_model(path):
    checkpoint = torch.load(path)
    try:
        features = checkpoint['features']
    except:
        print("Could not load number features from checkpoint. Use 256 features.") 
        features = 256
    pos_enc = checkpoint['pos_enc']
    model = EM_Simulator(pos_enc, large=True, features = features).cuda() # TODO EM_Simulator(self.pos_enc//2, large=True).cuda()
    try:
        model.load_state_dict(checkpoint['model_large_state_dict'])
    except:
        model.load_state_dict(checkpoint['model'])

    return model, pos_enc

def load_explicit_model(path):
    checkpoint = torch.load(path)
    recon = checkpoint['model_large_state_dict']
    return recon

def load_data_params(path):
    try:
        w, h, ray_length, center_distance, norm_coords = read_pickle(str(path)+"/params/_wh.pkl")
    except:
        w, h, ray_length = read_pickle(str(path)+"/params/_wh.pkl")
        norm_coords = False
        center_distance = ray_length
        print("No coordinate normalization applied")
    return w,h,ray_length, center_distance, norm_coords

def read_pickle(path):
    with gzip.open(path, 'rb') as f:
        p = pickle.Unpickler(f)
        data = p.load()
    return data

def min_max_torch(volume):
    return (volume - torch.min(volume))/(torch.max(volume)- torch.min(volume))

def min_max_np(volume):
    return (volume - np.min(volume))/(np.max(volume)- np.min(volume))

def min_max(volume):
    try: 
        vol = min_max_np(volume)
    except: 
        vol = min_max_torch(volume)
    return vol 



# SAVE FUNCTIONS
def save_tif(path, img):
        transformed_img = (img*((2**16)-1)).astype(np.uint16)
        img_out = Image.fromarray(transformed_img)
        img_out.save(path)
        return True

def save_tif_stack(volume,path):
    for i,slice in tqdm(enumerate(volume)):
        save_tif(path+"/Slice_"+str(i)+".tif", slice)


def save_raw(path, volume):
    path_large = path
    volume = volume*255
    if(np.max(volume)>255):
        volume[volume>255] = 255
    if(np.min(volume)<0):
        volume[volume<0] = 0
    volume = volume.astype('uint8')
    volume.tofile(path_large+"/volume.raw")
    print("Saved volume at: "+path_large+"/volume.raw")
    return

def save_txt(text, path):
    f = open(path, "w")
    f.write(text)
    f.close()
    print("Saved text file to: "+str(path))


# LOAD FUNCTIONS
def open_mrc(path):
    with mrcfile.open(path) as mrc:
        volume_data = mrc.data
    print("Max loaded data: "+str(np.max(volume_data)))
    return volume_data

def open_tif_stack(paths):
    img = Image.open(paths[0])
    img = np.array(img)
    volume = np.zeros((len(paths), *img.shape), dtype=np.float32)
    for i,path in enumerate(paths): 
        img = Image.open(path).convert("L") 
        img = np.array(img) 
        volume[i,:,:] = img
    print("Max loaded data: "+str(np.max(volume)))
    return volume

def open_raw(path, voxel_type='uint8'):
    try:
        data = np.fromfile(path[0], dtype=voxel_type)
    except: 
        try:
            data = np.fromfile(path, dtype=voxel_type)
        except:
            path = glob.glob(path+"/*.raw")
            data = np.fromfile(path[0], dtype=voxel_type)
            

    shape = int(round(data.shape[0]**(1/3),0))
    data = data.reshape((shape,shape,shape))
    print("Max loaded data: "+str(np.max(data)))
    return data



#METRICS
def psnr(vol_gt, vol, slices=False):
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

def mse(vol_gt, vol, slices=False):
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


def dssim(vol_gt, vol, slices = False):
    if(slices):
        mdssim = []
        for proj_gt, proj in zip(vol_gt, vol):
            ssim = m.structural_similarity(proj_gt, proj, channel_axis=False)
            mdssim.append((1-ssim)/2)
        std_dssim = np.std(mdssim)
        mdssim = np.mean(mdssim)
    else: 
        mssim = m.structural_similarity(vol_gt, vol, channel_axis=False)
        mdssim = (1-mssim)/2
        std_dssim = 0
    return mdssim, std_dssim
