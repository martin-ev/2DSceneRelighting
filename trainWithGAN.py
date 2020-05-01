import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid

from utils.storage import save_trained, load_trained
from utils.device import setup_device
from utils.losses import ReconstructionLoss, SceneLatentLoss, LightLatentLoss, GANLoss, FoolGANLoss
import utils.ssim as pytorch_ssim
from lpips_pytorch import lpips
import utils.tensorboard as tensorboard

from utils.dataset import InputTargetGroundtruthDataset, TRAIN_DATA_PATH, VALIDATION_DATA_PATH, SameScene, SameLightDirection, SameLightColor
from torch.utils.data import DataLoader

from models.swapModels import IlluminationSwapNet as SwapNet
#from models.swapModels import AnOtherSwapNet as SwapNet
from models.patchGan import NLayerDiscriminator

def write_images(dataset):
    writer.add_image('Visualization-%s/1-Input' %dataset, make_grid(input_image[: SHOWN_SAMPLES]), step)
    writer.add_image('Visualization-%s/2-Input-Light-latent' %dataset, make_grid(input_light_latent[: SHOWN_SAMPLES]), step)
    writer.add_image('Visualization-%s/3-Target' %dataset, make_grid(target_image[: SHOWN_SAMPLES]), step)
    writer.add_image('Visualization-%s/4-Target-Light-latent %dataset', make_grid(target_light_latent[: SHOWN_SAMPLES]), step)
    writer.add_image('Visualization-%s/5-Ground-truth' %dataset, make_grid(groundtruth_image[: SHOWN_SAMPLES]), step)
    writer.add_image('Visualization-%s/6-Ground-truth-Light-latent' %dataset, make_grid(groundtruth_light_latent[: SHOWN_SAMPLES]), step)
    writer.add_image('Visualization-%s/7-Relit' %dataset, make_grid(relit_image[: SHOWN_SAMPLES]), step)
    
def write_measures(dataset):
    writer.add_scalar('%s/Generator_loss' %dataset, test_generator_loss/TESTING_BATCHES, step)
    writer.add_scalar('%s/Discriminator_loss' %dataset, test_discriminator_loss/TESTING_BATCHES, step)
    writer.add_scalar('%s/Score' %dataset, test_score/TESTING_BATCHES, step)
    writer.add_scalar('%s/SSIM' %dataset, test_ssim/TESTING_BATCHES, step)
    writer.add_scalar('%s/LPIPS' %dataset, test_lpips/TESTING_BATCHES, step)


# Get used device
GPU_IDS = [3]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'MergedModelsAnOtherSwapNet'
TRAIN_BATCH_SIZE = 15
TRAIN_NUM_WORKERS = 8
TEST_BATCH_SIZE = 15
TEST_NUM_WORKERS = 8
SIZE = 256
TRAIN_DURATION = 60000

# Configure training objects
generator = SwapNet().to(device)
discriminator = NLayerDiscriminator().to(device)
optimizerG = torch.optim.Adam(generator.parameters(), weight_decay=0)
optimizerD = torch.optim.Adam(discriminator.parameters(), weight_decay=0)
generator.train()
discriminator.train()  


# Losses
reconstruction_loss = ReconstructionLoss().to(device)
scene_latent_loss = SceneLatentLoss().to(device)
light_latent_loss = LightLatentLoss().to(device)
gan_loss = GANLoss().to(device)
fool_gan_loss = FoolGANLoss().to(device)
ssim_loss = pytorch_ssim.SSIM

# Configure dataloader
train_dataset = InputTargetGroundtruthDataset(transform=transforms.Resize(SIZE),
                                              data_path=TRAIN_DATA_PATH, locations = ["scene_nordic_23"])#, locations=['scene_abandonned_city_54'], input_directions = ["S"], target_directions = ["N"], input_colors = ["2500"], target_colors = ["6500"])
test_dataset = InputTargetGroundtruthDataset(transform=transforms.Resize(SIZE), data_path=VALIDATION_DATA_PATH, locations = ["scene_city_24"])
test_light_dir = InputTargetGroundtruthDataset(transform=transforms.Resize(SIZE), data_path=VALIDATION_DATA_PATH, pairing_strategies = [SameScene, SameLightColor], locations = ["scene_city_24"])
test_light_color = InputTargetGroundtruthDataset(transform=transforms.Resize(SIZE), data_path=VALIDATION_DATA_PATH, pairing_strategies = [SameScene, SameLightDirection], locations = ["scene_city_24"])
test_scene = InputTargetGroundtruthDataset(transform=transforms.Resize(SIZE), data_path=VALIDATION_DATA_PATH, pairing_strategies = [SameLightDirection, SameLightColor], locations = ["scene_city_24"])


train_dataloader  = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=TRAIN_NUM_WORKERS)
test_dataloader  = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=TEST_NUM_WORKERS)
test_light_dir_dataloader = DataLoader(test_light_dir, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=TEST_NUM_WORKERS)
test_light_color_dataloader = DataLoader(test_light_color, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=TEST_NUM_WORKERS)
test_scene_dataloader = DataLoader(test_scene, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=TEST_NUM_WORKERS)
test_dataloaders = [test_dataloader, test_light_dir_dataloader, test_light_color_dataloader, test_scene_dataloader]

print(f'Dataset contains {len(train_dataset)}+{len(test_dataset)} samples.')
print(f'Running for {TRAIN_DURATION} batches.')

# Configure tensorboard
writer = tensorboard.setup_summary_writer(NAME)
tensorboard_process = tensorboard.start_tensorboard_process()
SHOWN_SAMPLES = 3
TESTING_FREQ = 100 # every how many batches model is tested and tensorboard is updated
TESTING_BATCHES = 10 # how many batches for testing
print(f'{SHOWN_SAMPLES} samples will be visualized every {TESTING_FREQ} batches.')

    
# Train loop
train_generator_loss, train_discriminator_loss, train_score = 0., 0., 0.
train_dataloader_iter = iter(train_dataloader)  
train_batches_counter = 0 
while train_batches_counter < TRAIN_DURATION :
    # Load batch
    with torch.no_grad():
        try:
            train_batch = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            train_batch = next(train_dataloader_iter)   
        input_image = train_batch[0][0]['image'].to(device)
        target_image = train_batch[0][1]['image'].to(device)
        groundtruth_image = train_batch[1]['image'].to(device)
        input_color = train_batch[0][0]['color'].to(device)
        target_color = train_batch[0][1]['color'].to(device)
        input_direction = train_batch[0][0]['direction'].to(device)
        target_direction = train_batch[0][1]['direction'].to(device)

    # Generator : Forward  
    output = generator(input_image, target_image, groundtruth_image)
    relit_image, \
    input_light_latent, target_light_latent, groundtruth_light_latent, \
    input_scene_latent, target_scene_latent, groundtruth_scene_latent = output  
    disc_out_fake = discriminator(relit_image)
    
    discriminator_loss = fool_gan_loss(disc_out_fake)
    generator_loss = reconstruction_loss(relit_image, groundtruth_image) + .5 * discriminator_loss
    train_generator_loss += generator_loss.item()
    train_discriminator_loss += discriminator_loss.item()
    train_score += reconstruction_loss(input_image, groundtruth_image).item() / reconstruction_loss(relit_image, groundtruth_image).item()    
    # Generator : Backward 
    optimizerG.zero_grad()
    generator.zero_grad()
    optimizerD.zero_grad()
    discriminator.zero_grad()
    generator_loss.backward() # use requires_grad = False for speed? Et pour enlever ces zero_grad en double!
    discriminator.zero_grad()
    optimizerD.zero_grad()
    optimizerG.step()
    
    
    # Discriminator : Forward  
    output = generator(input_image, target_image, groundtruth_image)
    relit_image, \
    input_light_latent, target_light_latent, groundtruth_light_latent, \
    input_scene_latent, target_scene_latent, groundtruth_scene_latent = output 
    disc_out_fake = discriminator(relit_image) 
    disc_out_real = discriminator(groundtruth_image)
    
    discriminator_loss = gan_loss(disc_out_fake, disc_out_real)
    train_discriminator_loss += discriminator_loss.item()
    
    
    # Discriminator : Backward 
    optimizerD.zero_grad()
    discriminator.zero_grad()
    optimizerG.zero_grad()
    generator.zero_grad()
    discriminator_loss.backward()
    generator.zero_grad()
    optimizerG.zero_grad()

    optimizerD.step()
    
    # Update train_batches_counter
    train_batches_counter += 1

    # If it is time to do so, test and visualize current progress
    step, modulo = divmod(train_batches_counter, TESTING_FREQ)
    if modulo == 0:
        with torch.no_grad():
            # Switch to eval mode
            generator.eval()
            discriminator.eval()
            
            # Visualize train
            write_images("train")
            writer.add_scalar('Train/Generator_loss', train_generator_loss/TESTING_FREQ, step)
            writer.add_scalar('Train/Discriminator_loss', train_discriminator_loss/TESTING_FREQ, step)
            writer.add_scalar('Train/Score', train_score/TESTING_FREQ, step)
            print ('TrainLoss:', train_generator_loss/TESTING_FREQ, 'TrainScore:', train_score/TESTING_FREQ)
            
            # Reset train scalars
            train_generator_loss, train_discriminator_loss, train_score = 0., 0., 0.
            
            # Evaluate loop
            for counter, d in enumerate(test_dataloaders):
                test_generator_loss, test_discriminator_loss, test_score, test_ssim, test_lpips = 0., 0., 0., 0., 0.
                testing_batches_counter = 0 
                test_dataloader_iter = iter(d)
                
                while testing_batches_counter < TESTING_BATCHES :
                    #Load batch
                    try:
                        test_batch = next(test_dataloader_iter)
                    except StopIteration:
                        test_dataloader_iter = iter(d)
                        test_batch = next(test_dataloader_iter)

                    input_image = test_batch[0][0]['image'].to(device)
                    target_image = test_batch[0][1]['image'].to(device)
                    groundtruth_image = test_batch[1]['image'].to(device)
                    input_color = test_batch[0][0]['color'].to(device)
                    target_color = test_batch[0][1]['color'].to(device)
                    input_direction = test_batch[0][0]['direction'].to(device)
                    target_direction = test_batch[0][1]['direction'].to(device)

                    # Forward
                    output = generator(input_image, target_image, groundtruth_image)
                    relit_image, \
                    input_light_latent, target_light_latent, groundtruth_light_latent, \
                    input_scene_latent, target_scene_latent, groundtruth_scene_latent = output  
                    disc_out_fake = discriminator(relit_image)
                    disc_out_real = discriminator(target_image) # better than groundtruth_image

                    generator_loss = reconstruction_loss(relit_image, groundtruth_image) + .5 * fool_gan_loss(disc_out_fake)
                    discriminator_loss = gan_loss(disc_out_fake, disc_out_real)

                    test_generator_loss += generator_loss.item()
                    test_discriminator_loss += discriminator_loss.item()
                    test_score += reconstruction_loss(input_image, groundtruth_image).item() / reconstruction_loss(relit_image, groundtruth_image).item()
                    test_ssim += ssim_loss(relit_image, groundtruth_image)
                    test_lpips += lpips(relit_image, groundtruth_image)
                    
                    # Update testing_batches_counter
                    testing_batches_counter += 1            
                # Visualize test
                write_images("Test - %d" %counter)
                write_measures("Test - %d" %counter)
                print ('TestLoss:', test_generator_loss/TESTING_BATCHES, 'TestScore:', test_score/TESTING_BATCHES)
            
            
            # Switch to train mode
            generator.train()
            discriminator.train()  

# Store trained model
storage.save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
