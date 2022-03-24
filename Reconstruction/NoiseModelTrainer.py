import argparse
import os
import datetime
import glob
from PIL import Image
import numpy as np
import sys
import time
import traceback
import logging
import torch
import wandb

###############################################
# Train Noise Model in supervised fashion     #
###############################################


from Models.NoiseModel import NoiseModel
from Dataloader import Data_Loader
from Utils import *

GT_PATH = "/gt/"
NOISY_PATH = "/noisy/"

SAVE_STATE = "/save_state/"
FILE_NAME = "training_state.pth"

#TODO set your own wandb account details
os.environ['WANDB_API_KEY'] = ""
os.environ['WANDB_ENTITY']= ""

class NoiseModelTrainer():
    def __init__(self, args, log_folder_path):
        self.args = args
        self.log_folder_path = log_folder_path

        self.gt_imgs, self.noisy_imgs = self.load_data()
        self.start_iteration = 0

        if(self.args.resume_training):
            try:
                checkpoint = torch.load(self.args.resume_training+SAVE_STATE+FILE_NAME)
                self.args.type = checkpoint['type']

                self.noise_model = NoiseModel(num_layers_cond = 4, num_layers = 4, cond_type=self.args.type) 
                self.noise_model.load_state_dict(checkpoint['noise_model'])
                self.noise_model.cuda()

                optim = checkpoint['optim_type']
                if(optim == "sgd"):
                    lr_init = 5e-7
                    lr_end = 5e-8
                    self.optimizer = torch.optim.SGD(params = self.noise_model.parameters(), 
                        weight_decay= 1e-3,
                        lr= 1.,
                        momentum=0.9)

                elif(optim == "adam"):
                    lr_init = 5e-5
                    lr_end = 5e-6
                    self.optimizer = torch.optim.Adam(params = self.noise_model.parameters(), lr=1.)

                else: 
                    print("ERROR:: Failed to load optimizer: "+str(optim))
                
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_iteration = checkpoint['start_iteration']

                # Learning rate scheduler.
                num_steps = int(self.args.iterations) 
                start_decay = int(2*num_steps/3) 
                decay_steps = num_steps - start_decay
                lr_muls = []

                for i in range(int(args.iterations)+1):
                    if i > start_decay:
                        lr_muls.append(((i-start_decay)/decay_steps)*(lr_end-lr_init) + lr_init)
                    else:
                        lr_muls.append(lr_init)
                lambda_lr = lambda epoch: lr_muls[epoch]
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
                
                
            except: 
                print(traceback.format_exc())
                print("ERROR:: Could not resume training from path: "+str(self.args.resume_training))
                sys.exit(-1)
        else: 
            self.noise_model = NoiseModel(num_layers_cond = 4, num_layers = 4, cond_type=self.args.type) # TODO test conv? 
            self.noise_model.cuda()

            if(self.args.optim == "sgd"):
                lr_init = 5e-7
                lr_end = 5e-8
                self.optimizer = torch.optim.SGD(params = self.noise_model.parameters(), 
                    weight_decay= 1e-3,
                    lr= 1.,
                    momentum=0.9)
            elif(self.args.optim == "adam"):
                lr_init = 5e-5
                lr_end = 5e-6
                self.optimizer = torch.optim.Adam(params = self.noise_model.parameters(), lr=1.)
            else: 
                print("ERROR:: Wrong optimizer definition.")
                sys.exit(-1)

            # Learning rate scheduler.
            num_steps = int(self.args.iterations) 
            start_decay = int(2*num_steps/3) 
            decay_steps = num_steps - start_decay
            lr_muls = []

            for i in range(int(args.iterations)+1):
                if i > start_decay:
                    lr_muls.append(((i-start_decay)/decay_steps)*(lr_end-lr_init) + lr_init)
                else:
                    lr_muls.append(lr_init)
            lambda_lr = lambda epoch: lr_muls[epoch]
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        
        logging.basicConfig(filename=log_folder_path+'/log.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

    

    def save_img(self, path, img):
        img = np.squeeze(img)
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(path)
        return

    def load_data(self):
        if(self.args.noisy_path == None):
            gt_path = self.args.data_path+GT_PATH
            noisy_path = self.args.data_path+NOISY_PATH
        else: 
            gt_path = self.args.data_path
            noisy_path = self.args.noisy_path
        
        dl = Data_Loader(gt_path, None, None)
        gt_path = np.array(glob.glob(gt_path+"*.pkl"))
        noisy_path = np.array(glob.glob(noisy_path+"*.pkl"))
        _, _, gt_imgs,_,_ = dl.load_from_path(gt_path) 
        _, _, noisy_imgs,_,_ = dl.load_from_path(noisy_path) 

        print("Before normalization")
        print("Clean | Max: "+str(np.max(gt_imgs))+" | Min: "+str(np.min(gt_imgs)))
        print("Noisy | Max: "+str(np.max(noisy_imgs))+" | Min: "+str(np.min(noisy_imgs)))
        print()

        logging.info("Before normalization")
        logging.info("Clean | Max: "+str(np.max(gt_imgs))+" | Min: "+str(np.min(gt_imgs)))
        logging.info("Noisy | Max: "+str(np.max(noisy_imgs))+" | Min: "+str(np.min(noisy_imgs)))
        logging.info("")

        if(self.args.combined_norm):
            gt_imgs, noisy_imgs = dl.combined_norm(gt_imgs, noisy_imgs)
        else: 
            gt_imgs, noisy_imgs = dl.independent_norm(gt_imgs, noisy_imgs)

        print("After normalization")
        print("Clean | Max: "+str(np.max(gt_imgs))+" | Min: "+str(np.min(gt_imgs)))
        print("Noisy | Max: "+str(np.max(noisy_imgs))+" | Min: "+str(np.min(noisy_imgs)))
        print()

        logging.info("After normalization")
        logging.info("Clean | Max: "+str(np.max(gt_imgs))+" | Min: "+str(np.min(gt_imgs)))
        logging.info("Noisy | Max: "+str(np.max(noisy_imgs))+" | Min: "+str(np.min(noisy_imgs)))
        logging.info("")

        return torch.Tensor(gt_imgs), torch.Tensor(noisy_imgs)

    def log_likelihood(self,log_probs):
        log_probs = torch.reshape(log_probs, (self.args.batch_size, -1))
        return -torch.mean(torch.sum(log_probs, axis=1))

    def get_batch(self):
        rand_patch_x = torch.randint(0, self.gt_imgs.shape[0]-self.args.patch_size, (self.args.batch_size,))
        rand_patch_y = torch.randint(0, self.gt_imgs.shape[1]-self.args.patch_size, (self.args.batch_size,))

        clean_patches = torch.zeros((self.args.batch_size, self.args.patch_size, self.args.patch_size))
        noisy_patches = torch.zeros((self.args.batch_size, self.args.patch_size, self.args.patch_size))

        for b in range(self.args.batch_size):
            clean_patches[b,:,:] = self.gt_imgs[rand_patch_x[b]:rand_patch_x[b]+self.args.patch_size, rand_patch_y[b]:rand_patch_y[b]+self.args.patch_size, 0]
            noisy_patches[b,:,:] = self.noisy_imgs[rand_patch_x[b]:rand_patch_x[b]+self.args.patch_size, rand_patch_y[b]:rand_patch_y[b]+self.args.patch_size, 0]
        return clean_patches.cuda(), noisy_patches.cuda()

    def training(self):
        avg_loss = 0
        iteration = self.start_iteration

        while(True):
            iteration += 1
            self.optimizer.zero_grad()

            clean_patches, noisy_patches = self.get_batch()

            # Compute the log probability.
            log_prob = self.noise_model.log_prob(
                noisy_patches, 
                clean_patches)
            loss = self.log_likelihood(log_prob)
            avg_loss += loss
             # log validation losses
            wandb.log({"loss": loss}, step = iteration)

            # Loss backward.
            loss.backward()

            # Change variables.
            self.optimizer.step()

            # update lr
            if(iteration < self.args.iterations):
                self.scheduler.step()

            if(iteration == self.args.iterations):
                break

            if(iteration % self.args.val_step == 0):
                print("Iteration: "+str(iteration)+"/ "+str(self.args.iterations))
                middle = self.gt_imgs.shape[1]//2
                clean = self.gt_imgs[:,middle-500:middle+500].cuda()

                avg_loss = avg_loss/self.args.val_step

                wandb.log({"Average loss": avg_loss}, step = iteration)
                noise_pred = self.noise_model.sample_noise(clean)

                noisy_img_pred = clean+ noise_pred
                noisy_img = self.noisy_imgs[:,middle-500:middle+500]

                wandb.log({'Prediction Noisy Image': wandb.Image(min_max(noisy_img_pred).squeeze()), 
                            'True Noisy Image': wandb.Image(min_max(noisy_img).squeeze()),
                            'Clean Image': wandb.Image(min_max(clean).squeeze()), 
                            }, step = iteration)

                noisy_img_pred = noisy_img_pred.detach().cpu().numpy()
                noisy_img = noisy_img.detach().cpu().numpy()

                self.save_img(self.log_folder_path+"Noisy_Pred.png", noisy_img_pred)
                self.save_img(self.log_folder_path+"Noisy.png", noisy_img)
                self.save_img(self.log_folder_path+"Clean.png", clean.cpu().numpy())

                #save model, optimizer
                torch.save({
                    'noise_model': self.noise_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    
                    'optim_type': self.args.optim,
                    'start_iteration': iteration,
                    'type': self.args.type,
                    }, self.log_folder_path+SAVE_STATE+FILE_NAME)
        return



if __name__ == "__main__":

    print("******************************")
    print("Train Noise Flow")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Train the NoiseFlow')
    parser.add_argument('--iterations', type = int, default=400000, help='Number of training iterations (default: 200.000)')
    parser.add_argument('--val_step', type = int, default=10000, help='Number of iterations, until validation (default: 10.000)')

    parser.add_argument('--batch_size', type = int, default=16, help='Number of patches per minibatch (default: 1)')
    parser.add_argument('--patch_size', type = int, default=16, help='Number of rays per patch (default: 16)')

    parser.add_argument('--resume_training', type = str, default='', help='Path where to resume the training from (default: "")')
    parser.add_argument('--log_path', type = str, default='./NoiseModelRuns/', help='Logging path for NoiseFlow Training run (default: ./NoiseModelRuns/)')
    parser.add_argument('--data_path', type = str, default='./Data/Fibril/', help='Directory of the (clean) training data. Needs to contain .pkl files (default: ./Data/Fibril/)')
    parser.add_argument('--noisy_path', type = str, default='./Data/Fibril/', help='Directory of the training data (default: ./Data/Fibril/)')

    parser.add_argument('--type', type = str, default="mlp", help='Type of noise flow - one of [mlp, conv] (default: mlp)')
    parser.add_argument('--optim', type = str, default="sgd", help='Type of optimizer - one of [adam, sgd] (default: sgd)')

    parser.add_argument("--combined_norm", default=False, action="store_true", help="Weather to use combined normalization over clean and noisy images (default: False)")

    #Wandb
    parser.add_argument('--project', type = str, default='NoiseModelTraining', help='WandB project name (default: NoiseModelTraining)')
    parser.add_argument('--run_name', type = str, default='', help='WandB run name (default: Test)')
    parser.add_argument('--run_notes', type = str, default='', help='WandB run notes (default: )')

    args = parser.parse_args()

    print("Parameters:")
    print(args)

    # log folder
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    log_folder_path = args.log_path+"/logs_"+date_time+"/"
    if not os.path.isdir(log_folder_path):
        os.makedirs(log_folder_path)
        try:
            os.mkdir(log_folder_path+SAVE_STATE)
        except:
            print("WARNING:: Could not build paths")

    os.environ['WANDB_NAME']= args.run_name
    os.environ['WANDB_NOTES']=args.run_notes
    os.environ['WANDB_PROJECT']= args.project

    wandb.init(config = args)


    # Save args params
    f = open(log_folder_path+"/args.txt", "w")
    f.write(str(vars(args)))
    f.close()


    nt = NoiseModelTrainer(args, log_folder_path)
    nt.training()


    
