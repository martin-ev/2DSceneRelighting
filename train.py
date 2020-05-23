import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from utils.storage import save_trained, load_trained
from utils.device import setup_device
from utils.losses import ReconstructionLoss, SceneLatentLoss, LightLatentLoss, GANLoss, FoolGANLoss, ColorPredictionLoss, DirectionPredictionLoss
from pytorch_ssim import SSIM
from lpips_pytorch import LPIPS
import utils.tensorboard as tensorboard
from utils.dataset import InputTargetGroundtruthDataset, TRAIN_DATA_PATH, VALIDATION_DATA_PATH, SameScene, SameLightDirection, SameLightColor
from torch.utils.data import DataLoader
from utils.metrics import psnr
from models.swapModels import IlluminationSwapNet, AnOtherSwapNet, SwapNet512x1x1, GroundtruthEnvmapSwapNet, IlluminationPredicter
from models.patchGan import NLayerDiscriminator
import pickle

def write_images(writer, header, step, inputs, input_light_latents, targets, target_light_latents, groundtruthes, groundtruth_light_latents, relits):
    writer.add_image(f'Visualization-{header}/1-Input', make_grid(inputs), step)
    writer.add_image(f'Visualization-{header}/2-Input-Light-latent', make_grid(input_light_latents), step)
    writer.add_image(f'Visualization-{header}/3-Target', make_grid(targets), step)
    writer.add_image(f'Visualization-{header}/4-Target-Light-latent', make_grid(target_light_latents), step)
    writer.add_image(f'Visualization-{header}/5-Ground-truth', make_grid(groundtruthes), step)
    writer.add_image(f'Visualization-{header}/6-Ground-truth-Light-latent', make_grid(groundtruth_light_latents), step)
    writer.add_image(f'Visualization-{header}/7-Relit', make_grid(relits), step)
    
def write_measures(writer, header, step,
                   generator_loss, discriminator_loss,
                   score, ssim, lpips, psnr,
                   scene_input_gt, scene_input_target, scene_gt_target,
                   light_input_gt, light_input_target, light_gt_target,
                   color_prediction, direction_prediction
                  ):                            
    writer.add_scalar(f'{header}/Generator_loss', generator_loss, step)
    writer.add_scalar(f'{header}/Discriminator_loss', discriminator_loss, step)
    writer.add_scalar(f'{header}/Score', score, step)
    writer.add_scalar(f'{header}/SSIM', ssim, step)
    writer.add_scalar(f'{header}/LPIPS', lpips, step)       
    writer.add_scalar(f'{header}/PSNR', psnr, step)
    writer.add_scalar(f'{header}/Distance scene latents (input, gt)', scene_input_gt, step)
    writer.add_scalar(f'{header}/Distance scene latents (input, target)', scene_input_target, step)
    writer.add_scalar(f'{header}/Distance scene latents (target, gt)', scene_gt_target, step)
    writer.add_scalar(f'{header}/Distance light latents (input, gt)', light_input_gt, step)
    writer.add_scalar(f'{header}/Distance light latents (input, target)', light_input_target, step)
    writer.add_scalar(f'{header}/Distance light latents (target, gt)', light_gt_target, step)
    writer.add_scalar(f'{header}/Color prediction error', color_prediction, step)
    writer.add_scalar(f'{header}/Direction prediction error', direction_prediction, step)
    
def get_generator_model(model_name):
    if model_name=="IlluminationSwapNet": 
        return IlluminationSwapNet
    if model_name=="AnOtherSwapNet": 
        return AnOtherSwapNet
    if model_name=="SwapNet512x1x1": 
        return SwapNet512x1x1
    if model_name=="GroundtruthEnvmapSwapNet": 
        return GroundtruthEnvmapSwapNet
def get_light_latent_size(model_name):
    if model_name=="IlluminationSwapNet": 
        return 1*512
    if model_name=="AnOtherSwapNet": 
        return 3*16*16
    if model_name=="SwapNet512x1x1": 
        return 1*4*4
    if model_name=="GroundtruthEnvmapSwapNet": 
        return 1*512
    
def next_batch(dataloader_iter, dataloader):
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
    return batch, dataloader_iter

def extract_from_batch(batch, device):
    input_image = batch[0][0]['image'].to(device)
    target_image = batch[0][1]['image'].to(device)
    groundtruth_image = batch[1]['image'].to(device)
    input_color = batch[0][0]['color'].to(device, dtype=torch.float32)
    target_color = batch[0][1]['color'].to(device, dtype=torch.float32)
    groundtruth_color = batch[1]['color'].to(device, dtype=torch.float32)
    input_direction = batch[0][0]['direction'].to(device, dtype=torch.float32)
    target_direction = batch[0][1]['direction'].to(device, dtype=torch.float32)
    groundtruth_direction = batch[1]['direction'].to(device, dtype=torch.float32)
    return (input_image, target_image, groundtruth_image,
            input_color, target_color, groundtruth_color,
            input_direction, target_direction, groundtruth_direction)

#https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
import subprocess
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

import gc
def get_memory_objects():
    #https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/2
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(list(obj.size()), obj.device)
        except:
            pass
    
    
def main(config):
    # Device to use
    device = setup_device(config["gpus"])
    
    # Configure training objects
    # Generator
    model_name = config["model"]
    generator = get_generator_model(model_name)().to(device)
    weight_decay = config["L2_regularization_generator"]
    if config["use_illumination_predicter"]:
        light_latent_size = get_light_latent_size(model_name)
        illumination_predicter = IlluminationPredicter(in_size=light_latent_size).to(device)
        optimizerG = torch.optim.Adam(list(generator.parameters()) + list(illumination_predicter.parameters()), weight_decay=weight_decay)
    else:
        optimizerG = torch.optim.Adam(generator.parameters(), weight_decay=weight_decay)
    # Discriminator
    if config["use_discriminator"]:
        if config["discriminator_everything_as_input"]:
            raise NotImplementedError #TODO
        else:
            discriminator = NLayerDiscriminator().to(device)
        optimizerD = torch.optim.Adam(discriminator.parameters(), weight_decay=config["L2_regularization_discriminator"])
        
    # Losses
    reconstruction_loss = ReconstructionLoss().to(device)
    if config["use_illumination_predicter"]:
        color_prediction_loss = ColorPredictionLoss().to(device)
        direction_prediction_loss = DirectionPredictionLoss().to(device)
    if config["use_discriminator"]:
        gan_loss = GANLoss().to(device)
        fool_gan_loss = FoolGANLoss().to(device)
   
    # Metrics
    if "scene_latent" in config["metrics"]:
        scene_latent_loss = SceneLatentLoss().to(device)
    if "light_latent" in config["metrics"]:
        light_latent_loss = LightLatentLoss().to(device)
    if "LPIPS" in config["metrics"]:
        lpips_loss = LPIPS(
            net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
            version='0.1'  # Currently, v0.1 is supported
        ).to(device)
    if "SSIM" in config["metrics"]:
        ssim_loss = SSIM().to(device)
        
    
    # Configure dataloader
    size = config['image_resize']
    # train
    try:
        file = open('traindataset'+str(config['overfit_test'])+str(size)+'.pickle','rb')
        print("Restoring train dataset from pickle file")
        train_dataset = pickle.load( file)
        file.close()
        print("Restored train dataset from pickle file")
    except:
        train_dataset = InputTargetGroundtruthDataset(transform=transforms.Resize(size),
                                                      data_path=TRAIN_DATA_PATH,
                                                      locations=['scene_abandonned_city_54'] if config['overfit_test'] else None,
                                                      input_directions = ["S", "E"] if config['overfit_test'] else None,
                                                      target_directions = ["S", "E"] if config['overfit_test'] else None,
                                                      input_colors = ["2500", "6500"] if config['overfit_test'] else None,
                                                      target_colors = ["2500", "6500"] if config['overfit_test'] else None)
        file = open("traindataset"+str(config['overfit_test'])+str(size)+'.pickle','wb')
        pickle.dump(train_dataset, file)
        file.close()
        print("saved traindataset"+str(config['overfit_test'])+str(size)+'.pickle')
    train_dataloader  = DataLoader(train_dataset, 
                                   batch_size=config['train_batch_size'],
                                   shuffle=config['shuffle_data'],
                                   num_workers=config['train_num_workers'])
    # test
    try:
        file = open("testdataset"+str(config['overfit_test'])+str(size)+'.pickle','rb')
        print("Restoring full test dataset from pickle file")
        test_dataset = pickle.load( file)
        file.close()
        print("Restored full test dataset from pickle file")
    except:
        test_dataset = InputTargetGroundtruthDataset(transform=transforms.Resize(size),
                                                     data_path=VALIDATION_DATA_PATH,
                                                     locations=["scene_city_24"] if config['overfit_test'] else None,
                                                     input_directions = ["S", "E"] if config['overfit_test'] else None,
                                                     target_directions = ["S", "E"] if config['overfit_test'] else None,
                                                     input_colors = ["2500", "6500"] if config['overfit_test'] else None,
                                                     target_colors = ["2500", "6500"] if config['overfit_test'] else None)
        file = open("testdataset"+str(config['overfit_test'])+str(size)+'.pickle','wb')
        pickle.dump(test_dataset, file)
        file.close()
        print("saved testdataset"+str(config['overfit_test'])+str(size)+'.pickle')
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['test_batch_size'],
                                 shuffle=config['shuffle_data'],
                                 num_workers=config['test_num_workers'])
    test_dataloaders = {"full": test_dataloader}
    if config["testing_on_subsets"]:
        additional_pairing_strategies = [[SameLightColor()],
                                         [SameLightDirection()],
                                         #[SameScene()],
                                         #[SameScene(), SameLightColor()],
                                         #[SameScene(), SameLightDirection()],
                                         #[SameLightDirection(), SameLightColor()],
                                         #[SameScene(), SameLightDirection(), SameLightColor()]
                                        ]
        for pairing_strategies in additional_pairing_strategies:
            try:
                file = open("testdataset"+str(config['overfit_test'])+str(size)+str(pairing_strategies)+'.pickle','rb')
                print("Restoring test dataset "+ str(pairing_strategies)+" from pickle file")
                test_dataset = pickle.load( file)
                file.close()
                print("Restored test dataset "+ str(pairing_strategies)+" from pickle file")
            except:
                test_dataset = InputTargetGroundtruthDataset(transform=transforms.Resize(size),
                                                             data_path=VALIDATION_DATA_PATH,
                                                             pairing_strategies = pairing_strategies,
                                                             locations=["scene_city_24"] if config['overfit_test'] else None,
                                                             input_directions = ["S", "E"] if config['overfit_test'] else None,
                                                             target_directions = ["S", "E"] if config['overfit_test'] else None,
                                                             input_colors = ["2500", "6500"] if config['overfit_test'] else None,
                                                             target_colors = ["2500", "6500"] if config['overfit_test'] else None)
                file = open("testdataset"+str(config['overfit_test'])+str(size)+str(pairing_strategies)+'.pickle','wb')
                pickle.dump(test_dataset, file)
                file.close()
                print("saved testdataset"+str(config['overfit_test'])+str(size)+str(pairing_strategies)+'.pickle')                
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=config['test_batch_size'],
                                         shuffle=config['shuffle_data'],
                                         num_workers=config['test_num_workers'])
            test_dataloaders[str(pairing_strategies)] = test_dataloader
    print(f'Dataset contains {len(train_dataset)} train samples and {len(test_dataset)} test samples.')
    print(f'{config["shown_samples_grid"]} samples will be visualized every {config["testing_frequence"]} batches.')
    print(f'Evaluation will be made every {config["testing_frequence"]} batches on {config["batches_for_testing"]} batches')
    
    # Configure tensorboard
    writer = tensorboard.setup_summary_writer(config['name'])
    tensorboard_process = tensorboard.start_tensorboard_process() #TODO: config["tensorboard_port"]    
        
    # Train loop
    
    # Init train scalars
    (train_generator_loss,
     train_discriminator_loss,
     train_score,
     train_lpips,
     train_ssim,
     train_psnr,
     train_scene_latent_loss_input_gt,
     train_scene_latent_loss_input_target,
     train_scene_latent_loss_gt_target,
     train_light_latent_loss_input_gt,
     train_light_latent_loss_input_target,
     train_light_latent_loss_gt_target,
     train_color_prediction_loss,
     train_direction_prediction_loss) = (0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.,
                                         0.)

    # Init train loop
    train_dataloader_iter = iter(train_dataloader)
    train_batches_counter = 0 
    print(f'Running for {config["train_duration"]} batches.')
    
    # Train loop
    while train_batches_counter < config['train_duration'] : 
        #with torch.autograd.detect_anomaly():

        # Load batch
        if config["debug"]: print('Load batch', get_gpu_memory_map())
        with torch.no_grad():
            train_batch, train_dataloader_iter = next_batch(train_dataloader_iter, train_dataloader)
            (input_image, target_image, groundtruth_image,
            input_color, target_color, groundtruth_color,
            input_direction, target_direction, groundtruth_direction) = extract_from_batch(train_batch, device)

        # Generator
        # Generator: Forward  
        if config["debug"]: print('Generator: Forward', get_gpu_memory_map())
        output = generator(input_image, target_image, groundtruth_image)
        (relit_image, 
        input_light_latent, target_light_latent, groundtruth_light_latent,
        input_scene_latent, target_scene_latent, groundtruth_scene_latent) = output 
        r = reconstruction_loss(relit_image, groundtruth_image)
        generator_loss = config['generator_loss_reconstruction_l2_factor'] * r
        if config["use_illumination_predicter"]:
            input_illumination = illumination_predicter(input_light_latent)
            target_illumination = illumination_predicter(target_light_latent)
            groundtruth_illumination = illumination_predicter(groundtruth_light_latent)
            c = (1/3 *  color_prediction_loss(input_illumination[:,0], input_color)
                 + 1/3 *  color_prediction_loss(target_illumination[:,0], target_color)
                 + 1/3 *  color_prediction_loss(groundtruth_illumination[:,0], groundtruth_color))
            d = (1/3 *  direction_prediction_loss(input_illumination[:,1], input_direction)
                 + 1/3 *  direction_prediction_loss(target_illumination[:,1], target_direction)
                 + 1/3 *  direction_prediction_loss(groundtruth_illumination[:,1], groundtruth_direction))
            generator_loss += config['generator_loss_color_l2_factor'] * c
            generator_loss += config['generator_loss_direction_l2_factor'] * d
            train_color_prediction_loss += c.item()
            train_direction_prediction_loss += d.item()
        train_generator_loss += generator_loss.item()
        train_score += reconstruction_loss(input_image, groundtruth_image).item() / reconstruction_loss(relit_image, groundtruth_image).item()
        if "scene_latent" in config["metrics"]:                            
            train_scene_latent_loss_input_gt += scene_latent_loss(input_image, groundtruth_image).item()
            train_scene_latent_loss_input_target += scene_latent_loss(input_image, target_image).item()
            train_scene_latent_loss_gt_target += scene_latent_loss(target_image, groundtruth_image).item()
        if "light_latent" in config["metrics"]:
            train_light_latent_loss_input_gt += light_latent_loss(input_image, groundtruth_image).item()
            train_light_latent_loss_input_target += light_latent_loss(input_image, target_image).item()
            train_light_latent_loss_gt_target += light_latent_loss(target_image, groundtruth_image).item()
        if "LPIPS" in config["metrics"]:
            train_lpips += lpips_loss(relit_image, groundtruth_image).item()
        if "SSIM" in config["metrics"]:
            train_ssim += ssim_loss(relit_image, groundtruth_image).item()
        if "PSNR" in config["metrics"]:
            train_psnr += psnr(relit_image, groundtruth_image).item()

        # Generator: Backward 
        if config["debug"]: print('Generator: Backward', get_gpu_memory_map())
        optimizerG.zero_grad()
        if config["use_discriminator"]: optimizerD.zero_grad()
        if config["use_discriminator"]: discriminator.zero_grad()
        generator_loss.backward() # use requires_grad = False for speed? Et pour enlever ces zero_grad en double!
        # Generator: Update parameters
        if config["debug"]: print('Generator: Update parameters', get_gpu_memory_map())
        optimizerG.step() 
        
        
        # Discriminator
        if config["use_discriminator"]:
            if config["debug"]: print('Discriminator', get_gpu_memory_map())
            # Discriminator : Forward  
            output = generator(input_image, target_image, groundtruth_image)
            (relit_image, 
            input_light_latent, target_light_latent, groundtruth_light_latent,
            input_scene_latent, target_scene_latent, groundtruth_scene_latent) = output 
            disc_out_fake = discriminator(relit_image) 
            disc_out_real = discriminator(groundtruth_image)
            discriminator_loss = config['discriminator_loss_gan_factor'] * gan_loss(disc_out_fake, disc_out_real)
            train_discriminator_loss += discriminator_loss.item()
            # Discriminator : Backward 
            optimizerD.zero_grad()
            discriminator.zero_grad()
            optimizerG.zero_grad()
            generator.zero_grad()
            discriminator_loss.backward()
            generator.zero_grad()
            optimizerG.zero_grad()
            # Discriminator : Update parameters
            optimizerD.step()
            
        # Update train_batches_counter
        train_batches_counter += 1
        
        # If it is time to do so, test and visualize current progress
        step, modulo = divmod(train_batches_counter, config['testing_frequence'])
        if modulo == 0:
            with torch.no_grad(): 
                
                # Visualize train
                if config["debug"]: print('Visualize train', get_gpu_memory_map())
                write_images(writer = writer,
                             header = "Train",
                             step = step,
                             inputs = input_image[:config['shown_samples_grid']],
                             input_light_latents = input_light_latent[:config['shown_samples_grid']],
                             targets = target_image[:config['shown_samples_grid']],
                             target_light_latents = target_light_latent[:config['shown_samples_grid']],
                             groundtruthes = groundtruth_image[:config['shown_samples_grid']],
                             groundtruth_light_latents = groundtruth_light_latent[:config['shown_samples_grid']],
                             relits = relit_image[:config['shown_samples_grid']])
                write_measures(writer = writer,
                               header = "Train",
                               step = step,
                               generator_loss = train_generator_loss/config['testing_frequence'],
                               discriminator_loss = train_discriminator_loss/config['testing_frequence'],
                               score = train_score/config['testing_frequence'],
                               ssim = train_ssim/config['testing_frequence'],
                               lpips = train_lpips/config['testing_frequence'],
                               psnr = train_psnr/config['testing_frequence'],
                               scene_input_gt = train_scene_latent_loss_input_gt/config['testing_frequence'],
                               scene_input_target = train_scene_latent_loss_input_target/config['testing_frequence'],
                               scene_gt_target = train_scene_latent_loss_gt_target/config['testing_frequence'],
                               light_input_gt = train_light_latent_loss_input_gt/config['testing_frequence'],
                               light_input_target = train_light_latent_loss_input_target/config['testing_frequence'],
                               light_gt_target = train_light_latent_loss_gt_target/config['testing_frequence'],
                               color_prediction = train_color_prediction_loss/config['testing_frequence'],
                               direction_prediction = train_direction_prediction_loss/config['testing_frequence'])
                print ('Train',
                       'Loss:', train_generator_loss/config['testing_frequence'],
                       'Score:', train_score/config['testing_frequence'])  
                if config["debug_memory"]: 
                    print(get_gpu_memory_map()) 
                    #del generator_loss
                    #torch.cuda.empty_cache()
                    #print(get_gpu_memory_map())

                # Reset train scalars
                if config["debug"]: print('Reset train scalars', get_gpu_memory_map())
                (train_generator_loss,
                 train_discriminator_loss,
                 train_score,
                 train_lpips,
                 train_ssim,
                 train_psnr,
                 train_scene_latent_loss_input_gt,
                 train_scene_latent_loss_input_target,
                 train_scene_latent_loss_gt_target,
                 train_light_latent_loss_input_gt,
                 train_light_latent_loss_input_target,
                 train_light_latent_loss_gt_target,
                 train_color_prediction_loss,
                 train_direction_prediction_loss) = (0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.,
                                                     0.)

                
                # Test loop 
                
                if config["debug"]: print('Test loop', get_gpu_memory_map())
                for header, test_dataloader in test_dataloaders.items():                    
                    
                    # Init test scalars
                    if config["debug"]: print('Init test scalars', get_gpu_memory_map())
                    (test_generator_loss,
                     test_discriminator_loss,
                     test_score,
                     test_lpips,
                     test_ssim,
                     test_psnr,
                     test_scene_latent_loss_input_gt,
                     test_scene_latent_loss_input_target,
                     test_scene_latent_loss_gt_target,
                     test_light_latent_loss_input_gt,
                     test_light_latent_loss_input_target,
                     test_light_latent_loss_gt_target,
                     test_color_prediction_loss,
                     test_direction_prediction_loss) = (0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.,
                                                        0.)
                        
                    # Init test loop
                    if config["debug"]: print('Init test loop', get_gpu_memory_map())
                    test_dataloader_iter = iter(test_dataloader)
                    testing_batches_counter = 0 

                    while testing_batches_counter < config['batches_for_testing'] :
                        
                        # Load batch  
                        if config["debug"]: print('Load batch', get_gpu_memory_map())                      
                        test_batch, test_dataloader_iter = next_batch(test_dataloader_iter, test_dataloader)
                        (input_image, target_image, groundtruth_image,
                        input_color, target_color, groundtruth_color,
                        input_direction, target_direction, groundtruth_direction) = extract_from_batch(test_batch, device)

                        # Forward
                        
                        # Generator
                        if config["debug"]: print('Generator', get_gpu_memory_map())    
                        output = generator(input_image, target_image, groundtruth_image)
                        (relit_image, 
                        input_light_latent, target_light_latent, groundtruth_light_latent,
                        input_scene_latent, target_scene_latent, groundtruth_scene_latent) = output 
                        r = reconstruction_loss(relit_image, groundtruth_image)
                        generator_loss = config['generator_loss_reconstruction_l2_factor'] * r
                        if config["use_illumination_predicter"]:
                            input_illumination = illumination_predicter(input_light_latent)
                            target_illumination = illumination_predicter(target_light_latent)
                            groundtruth_illumination = illumination_predicter(groundtruth_light_latent)
                            c = (1/3 *  color_prediction_loss(input_illumination[:,0], input_color)
                                 + 1/3 *  color_prediction_loss(target_illumination[:,0], target_color)
                                 + 1/3 *  color_prediction_loss(groundtruth_illumination[:,0], groundtruth_color))
                            d = (1/3 *  direction_prediction_loss(input_illumination[:,1], input_direction)
                                 + 1/3 *  direction_prediction_loss(target_illumination[:,1], target_direction)
                                 + 1/3 *  direction_prediction_loss(groundtruth_illumination[:,1], groundtruth_direction))
                            generator_loss += config['generator_loss_color_l2_factor'] * c
                            generator_loss += config['generator_loss_direction_l2_factor'] * d
                            test_color_prediction_loss += c.item()
                            test_direction_prediction_loss += d.item()
                        test_generator_loss += generator_loss.item()
                        test_score += reconstruction_loss(input_image, groundtruth_image).item() / reconstruction_loss(relit_image, groundtruth_image).item()
                        if "scene_latent" in config["metrics"]:                            
                            test_scene_latent_loss_input_gt += scene_latent_loss(input_image, groundtruth_image).item()
                            test_scene_latent_loss_input_target += scene_latent_loss(input_image, target_image).item()
                            test_scene_latent_loss_gt_target += scene_latent_loss(target_image, groundtruth_image).item()
                        if "light_latent" in config["metrics"]:
                            test_light_latent_loss_input_gt += light_latent_loss(input_image, groundtruth_image).item()
                            test_light_latent_loss_input_target += light_latent_loss(input_image, target_image).item()
                            test_light_latent_loss_gt_target += light_latent_loss(target_image, groundtruth_image).item()
                        if "LPIPS" in config["metrics"]:
                            test_lpips += lpips_loss(relit_image, groundtruth_image).item()
                        if "SSIM" in config["metrics"]:
                            test_ssim += ssim_loss(relit_image, groundtruth_image).item()
                        if "PSNR" in config["metrics"]:
                            test_psnr += psnr(relit_image, groundtruth_image).item()

                        # Discriminator
                        if config["debug"]: print('Discriminator', get_gpu_memory_map())    
                        if config["use_discriminator"]:
                            disc_out_fake = discriminator(relit_image) 
                            disc_out_real = discriminator(groundtruth_image)
                            discriminator_loss = config['discriminator_loss_gan_factor'] * gan_loss(disc_out_fake, disc_out_real)
                            test_discriminator_loss += discriminator_loss.item()
                        
                        # Update testing_batches_counter
                        if config["debug"]: print('Update testing_batches_counter', get_gpu_memory_map())   
                        testing_batches_counter += 1   
                        
                    # Visualize test
                    if config["debug"]: print('Visualize test', get_gpu_memory_map()) 
                    write_images(writer = writer,
                                 header = "Test-"+header,
                                 step = step,
                                 inputs = input_image[:config['shown_samples_grid']],
                                 input_light_latents = input_light_latent[:config['shown_samples_grid']],
                                 targets = target_image[:config['shown_samples_grid']],
                                 target_light_latents = target_light_latent[:config['shown_samples_grid']],
                                 groundtruthes = groundtruth_image[:config['shown_samples_grid']],
                                 groundtruth_light_latents = groundtruth_light_latent[:config['shown_samples_grid']],
                                 relits = relit_image[:config['shown_samples_grid']])
                    write_measures(writer = writer,
                                   header = "Test-"+header,
                                   step = step,
                                   generator_loss = test_generator_loss/config['batches_for_testing'],
                                   discriminator_loss = test_discriminator_loss/config['batches_for_testing'],
                                   score = test_score/config['batches_for_testing'],
                                   ssim = test_ssim/config['batches_for_testing'],
                                   lpips = test_lpips/config['batches_for_testing'],
                                   psnr = test_psnr/config['batches_for_testing'],
                                   scene_input_gt = test_scene_latent_loss_input_gt/config['batches_for_testing'],
                                   scene_input_target = test_scene_latent_loss_input_target/config['batches_for_testing'],
                                   scene_gt_target = test_scene_latent_loss_gt_target/config['batches_for_testing'],
                                   light_input_gt = test_light_latent_loss_input_gt/config['batches_for_testing'],
                                   light_input_target = test_light_latent_loss_input_target/config['batches_for_testing'],
                                   light_gt_target = test_light_latent_loss_gt_target/config['batches_for_testing'],
                                   color_prediction = test_color_prediction_loss/config['batches_for_testing'],
                                   direction_prediction = test_direction_prediction_loss/config['batches_for_testing'])
                    print ('Test-'+header,
                           'Loss:', test_generator_loss/config['testing_frequence'],
                           'Score:', test_score/config['testing_frequence'])  
                    
                    if config["debug_memory"]:
                        print(get_gpu_memory_map()) 
                        #del generator_loss
                        #torch.cuda.empty_cache()
                        #print(get_gpu_memory_map())
    
if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Train one 2D scene relighting model with a configuration file.')
    parser.add_argument('-c', '--config', default='default_config.txt', type=str, help='configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as json_file:
        config = json.load(json_file)  
    main(config)