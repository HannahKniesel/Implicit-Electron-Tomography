import logging
import wandb
import torch
import datetime, time
import skimage.metrics as metric
import argparse
import os

from Trainer import Trainer
from Models.EMSimulator import EM_Simulator
from Models.NoiseModel import NoiseModel

from Utils import *
from Models.Utils_ModelSaveLoad import *

SAVE_STATE = "/save_state/"

###############################################
# Train Implicit Model for reconstruction     #
###############################################

#TODO set your own wandb account details
os.environ['WANDB_API_KEY'] = ""
os.environ['WANDB_ENTITY']=""


class Trainer_Implicit(Trainer):
    def __init__(self, args, log_folder_path, gt_path):
        args.grid_res = 0
        super().__init__(args, log_folder_path, gt_path)
        
        # initialize implicit models
        self.simulator_model = EM_Simulator(self.args.pos_enc, large=True, features = self.args.features).cuda()
        self.simulator_approximation = EM_Simulator(self.args.pos_enc//2, large=False, features = self.args.features).cuda()
        small_model_params = self.simulator_approximation.parameters()
        model_params = self.simulator_model.parameters()
        if(self.args.resume_training and self.ckpt): 
            self.simulator_approximation.load_state_dict(self.ckpt['model_small_state_dict'])
            self.simulator_model.load_state_dict(self.ckpt['model_large_state_dict'])


        #optimizer and scheduler
        self.optimizer, self.scheduler = self.setup_optimizer(model_params, self.args.optim)
        self.optimizer_approximation, self.scheduler_approximation = self.setup_optimizer(small_model_params, self.args.optim)


        #noise model
        if(self.args.loss == "mle"):
            self.noise_model = NoiseModel(num_layers_cond = self.args.nf_cond, num_layers = self.args.nf_noncond, cond_type='mlp') 
            self.noise_model.cuda()
            
            # for supervised training
            if(self.args.nf_path):
                checkpoint = torch.load(self.args.nf_path)
                print("INFO:: Load Noise Model from: "+str(self.args.nf_path))
                self.noise_model.load_state_dict(checkpoint['noise_model'])
                self.noise_model.eval()
                for param in self.noise_model.parameters():
                    param.requires_grad = False

            else: 
                print("INFO:: Start E2E Training.")
                if(self.args.resume_training and self.ckpt): 
                    self.noise_model.load_state_dict(self.ckpt['noise_model'])
                    print("Load noise model parameters")

                self.nf_optim, self.scheduler_nf = self.setup_optimizer(self.noise_model.parameters(), self.args.nf_optim)
                
        else: 
            self.noise_model = None


    def training(self):
        logging.info("Start Training reconstruction network.")
        logging.info("**************************************\n\n")

        samples_coarse = self.args.samples
        samples_fine = 2*self.args.samples

        loss_l2 = torch.nn.MSELoss()
        loss_l1 = torch.nn.L1Loss()
        
        l2_optim_small = torch.optim.Adam(self.simulator_approximation.parameters(), lr=self.args.lr)
        l2_optim = torch.optim.Adam(self.simulator_model.parameters() , lr=self.args.lr)

        last_psnr = 0
        best_psnr = 0
        reg_loss = 0
        mean_training_loss = 0
        iteration = self.start_iteration
        iter_time = 0
        last_saved_model = 0
        
        while(True):
            if(not self.args.convergence and iteration > self.args.iterations):
                break
            iteration = iteration+1
            start_time_iter = time.perf_counter()

            self.optimizer.zero_grad()
            self.optimizer_approximation.zero_grad()
            iter_loss_small = 0
            iter_loss_large = 0
            loss = 0

            #small network
            for i in range(self.args.accumulate):
                if(self.args.defocus):
                    pos_patches, dir_patches, label_patches = self.get_batch_data_training_rays_jitter()
                else: 
                    pos_patches, dir_patches, label_patches = self.get_batch_data_training_rays()
                
                num_patches, width, height, _ = pos_patches.shape
                pos_patches = pos_patches.reshape((-1,pos_patches.shape[-1]))
                dir_patches = dir_patches.reshape((-1,dir_patches.shape[-1]))
                samples = self.data_loader.sample_uniformly(pos_patches, dir_patches, samples_coarse)  
                pixel_values, densities = self.simulator_approximation(samples, get_single_densities=True)
                pixel_values = pixel_values.reshape((num_patches, width, height,1))

                if(self.args.loss == "l2"):
                    loss = loss_l2(pixel_values, label_patches)
                elif(self.args.loss == "l1"):
                    loss = loss_l1(pixel_values, label_patches)
                elif(self.args.loss == "mle"):
                    if(iteration < self.args.l2_iters):
                        loss = loss_l2(pixel_values, label_patches)
                    else: 
                        loss = self.mle_loss(self.noise_model, pixel_values, label_patches)

                
                if(self.args.tv):
                    tv = torch.mean(torch.abs(torch.diff(self.simulator_approximation.grid, dim = 0))) + torch.mean(torch.abs(torch.diff(self.simulator_approximation.grid, dim = 1))) + torch.mean(torch.abs(torch.diff(self.simulator_approximation.grid, dim = 2)))
                    reg_loss = self.args.tv*tv
                
                loss = (loss+reg_loss)/self.args.accumulate
                
                loss.backward()
                
                iter_loss_small += loss.item()
            
            if(iteration < self.args.l2_iters):
                l2_optim_small.step()
            else:
                self.optimizer_approximation.step()
                if(self.scheduler_approximation and iteration < self.args.iterations):
                    self.scheduler_approximation.step()
            wandb.log({"Loss/loss small": iter_loss_small/(self.args.accumulate)}, step = iteration)
            #large network
            for i in range(self.args.accumulate):
                if(self.args.defocus):
                    pos_patches, dir_patches, label_patches = self.get_batch_data_training_rays_jitter()
                else: 
                    pos_patches, dir_patches, label_patches = self.get_batch_data_training_rays()
               
                num_patches, width, height, _ = pos_patches.shape
                pos_patches = pos_patches.reshape((-1,pos_patches.shape[-1]))
                dir_patches = dir_patches.reshape((-1,dir_patches.shape[-1]))
                samples = self.data_loader.sample_uniformly(pos_patches, dir_patches, samples_coarse)

                _, densities = self.simulator_approximation(samples, get_single_densities=True)                
                samples = self.data_loader.inverse_transform_sampling(densities.detach(), pos_patches, dir_patches, samples_coarse, samples_fine)

                pixel_values, densities = self.simulator_model(samples, get_single_densities = True)
                pixel_values = pixel_values.reshape((num_patches, width, height,1))          
               
                if(self.args.loss == "l2"):
                    loss = loss_l2(pixel_values, label_patches)
                elif(self.args.loss == "l1"):
                    loss = loss_l1(pixel_values, label_patches)
                elif(self.args.loss == "mle"):
                    if(iteration < self.args.l2_iters):
                        loss = loss_l2(pixel_values, label_patches)
                    else: 
                        loss = self.mle_loss(self.noise_model, pixel_values, label_patches)

                
                if(self.args.tv):
                    tv = torch.mean(torch.abs(torch.diff(self.simulator_model.grid, dim = 0))) + torch.mean(torch.abs(torch.diff(self.simulator_model.grid, dim = 1))) + torch.mean(torch.abs(torch.diff(self.simulator_model.grid, dim = 2)))
                    reg_loss = self.args.tv*tv

               
                loss = (loss+reg_loss)/self.args.accumulate

                loss.backward()

                if(self.noise_model):
                    torch.nn.utils.clip_grad_norm_(self.noise_model.parameters(), 100.)
                
                iter_loss_large += loss.item()
            
            if(iteration < self.args.l2_iters):
                # Starting iterations
                l2_optim.step()
            else: 
                # Recon Optim
                self.optimizer.step()
                # Recon scheduler
                if(iteration < self.args.iterations):
                    self.scheduler.step()
                # Noise model Scheduler
                if(self.args.loss == "mle" and (not self.args.nf_path)): 
                    self.nf_optim.step()
                    if(iteration < self.args.iterations):
                        self.scheduler_nf.step()

            wandb.log({"Loss/loss large": iter_loss_large/(self.args.accumulate)}, step = iteration)

            mean_training_loss+=iter_loss_large/self.args.accumulate

            iter_time += time.perf_counter()-start_time_iter
            
            # print("Time for Iteration "+str(iteration)+": "+str(iter_time/iteration)+" Loss small: "+str(iter_loss_small)+" Loss large: "+str(iter_loss_large)+ " Regularization loss: "+str(reg_loss))

            if((iteration-1) % int(self.args.val_step) == 0):
                print("VALIDATION Iteration: "+str(iteration))
                iter_time = iter_time/self.args.val_step
                time_left = (self.args.iterations - iteration) * iter_time
                approx_end = datetime.datetime.now() +datetime.timedelta(0,time_left)    
                date_time = approx_end.strftime("%d.%m.%Y  %H:%M:%S")
                iter_time = 0

                print("Approximated end: "+str(date_time))

                img_prediction, img_prediction_small, img_labels = self.validation(self.simulator_model, self.simulator_approximation)

                if(self.args.loss == "mle" and (not self.args.nf_path)):
                    noisy_proj, nll, noise = self.validate_nf(self.noise_model)
                    noisy_proj_recon = img_prediction + self.noise_model.sample_noise(torch.from_numpy(img_prediction.astype(np.float32)).cuda()).detach().cpu().numpy()

                    # log Losses
                    wandb.log({"Noise/NLL Noise Flow": nll,
                                "Noise/Mean Noise": torch.mean(noise),
                                "Noise/STD Noise": torch.std(noise)},
                                step = iteration)
                    
                    wandb.log({"Noisy Label": wandb.Image(min_max(self.noisy_label.squeeze().reshape(self.data_loader.w, -1))), 
                            "Noisy Prediction from Real": wandb.Image(min_max(noisy_proj).reshape(self.data_loader.w, -1)),
                            "Noisy Prediction from Recon": wandb.Image(min_max(noisy_proj_recon).squeeze()), 
                            "Noise": wandb.Image(min_max(noise).squeeze())
                            }, step = iteration)


                wandb.log({"Loss/Mean training loss": mean_training_loss/self.args.val_step}, step = iteration)
                mean_training_loss = 0

                psnr = 0
                ssim = 0
                loss = 0

                psnr_small = 0
                ssim_small = 0
                loss_small = 0
                compare_to = np.squeeze(self.gt_val, axis = -1) 

                int_img = (min_max(compare_to)*255).astype(np.uint8).squeeze()
                int_pred = (min_max(img_prediction)*255).astype(np.uint8).squeeze()
                int_pred_small = (min_max(img_prediction_small)*255).astype(np.uint8).squeeze()

                if(len(int_img.shape)<3): 
                    int_img = np.expand_dims(int_img, axis = 0)
                    int_pred = np.expand_dims(int_pred, axis = 0)
                    int_pred_small = np.expand_dims(int_pred_small, axis = 0)

                max_iters = np.min((self.args.num_validation, len(self.val_files)))
                for i in range(max_iters):
                    psnr += metric.peak_signal_noise_ratio(int_img[i], int_pred[i,:,:])
                    ssim += metric.structural_similarity(int_img[i], int_pred[i,:,:])
                    loss += loss_l2(torch.Tensor(int_img[i]), torch.Tensor(int_pred[i,:,:])) 
                    
                    psnr_small += metric.peak_signal_noise_ratio(int_img[i], int_pred_small[i,:,:])
                    ssim_small += metric.structural_similarity(int_img[i], int_pred_small[i,:,:])
                    loss_small += loss_l2(torch.Tensor(int_img[i]), torch.Tensor(int_pred_small[i,:,:]))

                self.save_img(self.log_folder_path+"GT.png", compare_to)
                
                psnr = psnr/max_iters
                ssim = ssim/max_iters
                loss = loss/max_iters

                psnr_small = psnr_small/max_iters
                ssim_small = ssim_small/max_iters
                loss_small = loss_small/max_iters

                loss_diff = np.abs(last_psnr-psnr)
                last_psnr = psnr

                logging.info("\nIteration: "+str(iteration))

                logging.info("PSNR small: "+str(psnr_small))
                logging.info("PSNR large: "+str(psnr))

                logging.info("SSIM small: "+str(ssim_small))
                logging.info("SSIM large: "+str(ssim))

                logging.info("Loss difference: "+str(loss_diff))


                print("PSNR small: "+str(psnr_small))
                print("PSNR large: "+str(psnr))

                print("SSIM small: "+str(ssim_small))
                print("SSIM large: "+str(ssim))

                print("Loss difference: "+str(loss_diff))
                print()

                # log validation losses
                wandb.log({"Projections/loss": loss, 
                            "Projections/psnr": psnr,
                            "Projections/ssim": ssim, 
                            "Projections/loss small": loss_small,
                            "Projections/psnr small": psnr_small,
                            "Projections/ssim small": ssim_small
                            }, step = iteration)

                img_prediction =  torch.Tensor(img_prediction)
                img_prediction_small =  torch.Tensor(img_prediction_small)
                img_labels = torch.Tensor(img_labels)
               
                wandb.log({"Clean Label": wandb.Image(min_max(self.gt_val).squeeze()), 
                            "Large Prediction": wandb.Image(min_max(img_prediction).squeeze()),
                            "Small Prediction": wandb.Image(min_max(img_prediction_small).squeeze()), 
                            }, step = iteration)

                #Validate volume slice
                if(self.gt_vol is not None):
                    recon = self.predict_slice()
                    psnr_val, mse_val, dssim_val = self.validate_volumeslice(self.gt_vol, recon)
                    wandb.log({"Reconstruction Slice": wandb.Image(min_max(recon)),
                                "GT Slice": wandb.Image(min_max(self.gt_vol))
                                }, step = iteration)                
                    wandb.log({ "Volume/psnr_volume": psnr_val,
                                "Volume/mse_volume": mse_val, 
                                "Volume/dssim_volume": dssim_val,
                                }, step = iteration)
                else: 
                    recon = self.predict_slice()
                    wandb.log({"Reconstruction Slice": wandb.Image(min_max(recon))}, step = iteration)                


                # Save models and parameters
                if((psnr > best_psnr) or iteration == self.args.val_step):
                    last_saved_model = 0
                    best_psnr = psnr
                    print("INFO:: Saved new model parameters with validation PSNR: "+str(psnr))
                    dict_saveparams = {}
                    set_param(iteration, 'start_iteration', dict_saveparams)
                    set_param(self.args.convergence, 'convergence', dict_saveparams)
                    set_param(self.args.batch_size, 'batch_size', dict_saveparams)
                    set_param(self.args.accumulate, 'accumulate', dict_saveparams)
                    set_param(self.args.patch_size, 'patch_size', dict_saveparams)
                    set_param(self.args.num_validation, 'num_validation', dict_saveparams)
                    
                    set_param(self.args.pos_enc, 'pos_enc', dict_saveparams)
                    set_param(self.args.samples, 'samples', dict_saveparams)
                    set_param(self.args.features, 'features', dict_saveparams)

                    set_param(self.args.nf_cond, 'nf_cond', dict_saveparams)
                    set_param(self.args.nf_noncond, 'nf_noncond', dict_saveparams)

                    set_param(self.args.loss, 'loss', dict_saveparams)
                    set_param(self.args.tv, 'tv', dict_saveparams)
                    set_param(self.args.nf_optim, 'nf_optim', dict_saveparams)
                    set_param(self.args.nf_lr, 'nf_lr', dict_saveparams)
                    set_param(self.args.nf_path, 'nf_path', dict_saveparams)
                    set_param(self.args.optim, 'optim', dict_saveparams)
                    set_param(self.args.lr, 'lr', dict_saveparams)

                    set_param(self.args.lr_decay_start, 'lr_decay_start', dict_saveparams)
                    set_param(self.args.lr_decay_rate, 'lr_decay_rate', dict_saveparams)

                    set_param(self.data_loader.center_distance, 'center_distance', dict_saveparams)
                    set_param(self.data_loader.is_norm_coords, 'is_norm_coords', dict_saveparams)

                    set_param(self.args.defocus, 'defocus', dict_saveparams)
                    set_param(self.args.max_defocus, 'max_defocus', dict_saveparams)
                    set_param(self.args.std, 'std', dict_saveparams)


                    print("Save Dict: "+str(dict_saveparams))

                  
                    set_param(self.simulator_approximation.state_dict(), 'model_small_state_dict', dict_saveparams)
                    set_param(self.simulator_model.state_dict(), 'model_large_state_dict', dict_saveparams)
                    
                    if(self.noise_model):
                        set_param(self.noise_model.state_dict(), 'noise_model', dict_saveparams)

                    save_dict(dict_saveparams, self.log_folder_path+SAVE_STATE+"/training_state.pth")
                    dict_saveparams.clear()
                else: 
                    last_saved_model += 1

            if((last_saved_model>10) and (iteration > 100000)):
                print("Early stopping. Stop based on convergence.")
                break
                
        return

    def predict_slice(self):
        coords = make_coords_central_slice(self.val_shape_vol)
        
        if(self.data_loader.is_norm_coords):
            r = self.data_loader.center_distance
            norm = np.sqrt(1+(r**2))
            coords[:,:,1:] = (coords[:,:,1:]/norm)
        
        coords = coords.reshape(-1,3)
        bs = 1024
        i = 0
        prediction = np.zeros((self.val_shape_vol*self.val_shape_vol))
        while((i+bs)< coords.shape[0]):
            batch = coords[i:(i+bs), :]
            batch = pos_enc_fct(self.args.pos_enc, batch).cuda()
            prediction_batch = self.simulator_model(batch, is_training=False)
            prediction[i:(i+bs)] = prediction_batch.detach().cpu().numpy().squeeze()
            i = i+bs
        
        if(i < coords.shape[0]):
            batch = coords[i:,:]
            batch = pos_enc_fct(self.args.pos_enc, batch).cuda()
            prediction_batch = self.simulator_model(batch, is_training=False)
            prediction[i:(i+bs)] = prediction_batch.detach().cpu().numpy().squeeze()

        prediction = prediction.reshape(self.val_shape_vol, self.val_shape_vol)
        prediction = np.rot90(prediction, k=3)
        # prediction = prediction[:,::-1]
        return prediction
        

if __name__ == "__main__":

    print("******************************")
    print("Train Reconstruction Network")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Train the reconstruction network')
    
    #Training Parameters
    parser.add_argument('--iterations', type = int, default=400000, help='Number of training iterations (default: 400.000)')
    parser.add_argument('--lr_decay_start', type = int, default=100000, help='Start Iteration for LR decay (default: 100.000)')
    parser.add_argument('--lr_decay_rate', type = int, default=100, help='Factor by which the lr is decaying. (default: 100)')

    parser.add_argument('--val_step', type = int, default=10000, help='Number of iterations, until validation (default: 10.000)')
    parser.add_argument('--convergence', action="store_true", default=False, help='Train until convergence (default: False)')
    parser.add_argument('--batch_size', type = int, default=2, help='Number of patches per minibatch (default: 2)')
    parser.add_argument('--accumulate', type = int, default=4, help='Number of accumulating gradients (default: 4)')
    parser.add_argument('--patch_size', type = int, default=16, help='Number of rays per patch (default: 16)')
    parser.add_argument('--num_validation', type = int, default=1, help='Number of images to use for validation (default: 1)')

    parser.add_argument('--resume_training', type = str, default='', help='Path where to resume the training from (default: "")')

    #Logging and Data
    parser.add_argument('--log_path', type = str, default='./TrainingRuns/', help='Logging path for Training run (default: ./TrainingRuns/)')
    parser.add_argument('--data_path', type = str, default='./TrainingData/', help='Directory of the training data (default: ./TrainingData/)')
    parser.add_argument('--gt_path', type = str, default='', help='Directory of ground truth data (default: "")')
    parser.add_argument('--gt_vol', type = str, default='./Tomograms/', help='Directory of GT Phantom volume. (default: ./Tomograms/)')

    #Implicit model 
    parser.add_argument('--pos_enc', type = int, default=10, help='Positional Encoding (default: 10)')
    parser.add_argument('--samples', type = int, default=64, help='Number of samples along a ray (default: 64)')
    parser.add_argument('--features', type = int, default=256, help='Number features of hidden layer in implicit model (default: 256)')

    #Noise model
    parser.add_argument('--nf_noncond', type = int, default=4, help='Number non conditional layers noise model (default: 4)')
    parser.add_argument('--nf_cond', type = int, default=4, help='Number conditional layers noise model (default: 4)')
    
    #Optimizer
    parser.add_argument('--loss', type = str, default='l2', help='Loss to use - one of [mle, l2, l1] (default: l2)')
    parser.add_argument('--tv', type = float, default='0', help='weight to use TV 1D regularization (default: 0)')
    parser.add_argument('--nf_optim', type = str, default='sgd', help='Optimizer (NF) to use - one of [adam, sgd] (default: adam)')
    parser.add_argument('--nf_lr', type = float, default=5e-5, help='Path where to load noise flow from (default: "")')
    parser.add_argument('--nf_path', type = str, default='', help='Path where to load noise flow from (default: "")')
    parser.add_argument('--optim', type = str, default='adam', help='Optimizer to use - one of [adam, sgd] (default: adam)')
    parser.add_argument('--lr', type = float, default=5e-5, help='Learning rate (default: 5e-5)')

    #Wandb
    parser.add_argument('--project', type = str, default='EM_Tomogram', help='WandB project name (default: EM_Tomogram_Grid)')
    parser.add_argument('--run_name', type = str, default='', help='WandB run name (default: Test)')
    parser.add_argument('--run_notes', type = str, default='', help='WandB run notes (default: )')

    #Starting iterations
    parser.add_argument('--l2_iters', type = int, default=100, help='L2 only iterations (default: 100)')

    #defocus
    parser.add_argument('--defocus', action="store_true", default=False, help='Take defocus into account during training (default: False)')
    parser.add_argument('--max_defocus', type = int, default=50, help='Maximum defocus in tilt series (default: 50)')
    parser.add_argument('--std', type = float, default=10, help='Std of gaussian for blur (default: 10)')

    parser.add_argument('--sweep', type = str, default='false', help='Do not set wandb project parameter, since sweep is running')

    args = parser.parse_args()

    print("Parameters:")
    print(args)

    gt_path = None
    if(args.gt_path != ""):
        gt_path = args.gt_path

    # log folder
    now = datetime.datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    log_folder_path = args.log_path+"/logs_"+date_time+"/"
    if not os.path.isdir(log_folder_path):
        os.makedirs(log_folder_path)

    # Save args params
    f = open(log_folder_path+"/args.txt", "w")
    f.write(str(vars(args)))
    f.close()

    #wandb
    if(args.sweep == "false"):
        os.environ['WANDB_NAME']= args.run_name
        os.environ['WANDB_NOTES']=args.run_notes
        os.environ['WANDB_PROJECT']= args.project
    
    wandb.init(config = args)

    t = Trainer_Implicit(args, log_folder_path, gt_path)
    t.training()