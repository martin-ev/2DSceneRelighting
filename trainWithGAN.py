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
import utils.tensorboard as tensorboard

from utils.dataset import InputTargetGroundtruthDataset, TRAIN_DATA_PATH, VALIDATION_DATA_PATH
from torch.utils.data import DataLoader

from models.swapModels import IlluminationSwapNet as SwapNet
#from models.swapModels import AnOtherSwapNet as SwapNet
from models.patchGan import NLayerDiscriminator


# Get used device
GPU_IDS = [2]
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

# Configure dataloader
train_dataset = InputTargetGroundtruthDataset(transform=transforms.Resize(SIZE),
                                              data_path=TRAIN_DATA_PATH)
#locations=['scene_abandonned_city_54'], input_directions = ["S"], target_directions = ["N"], input_colors = ["2500"], target_colors = ["6500"]
test_dataset = InputTargetGroundtruthDataset(transform=transforms.Resize(SIZE),
                                             data_path=VALIDATION_DATA_PATH)
train_dataloader  = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=TRAIN_NUM_WORKERS)
test_dataloader  = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=TEST_NUM_WORKERS)
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
test_dataloader_iter = iter(test_dataloader)
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
    generator_loss = reconstruction_loss(relit_image, groundtruth_image) + .3 * fool_gan_loss(disc_out_fake)
    train_generator_loss += generator_loss.item()
    train_discriminator_loss += generator_loss.item()
    train_score += reconstruction_loss(input_image, groundtruth_image).item() / reconstruction_loss(relit_image, groundtruth_image).item()    
    # Generator : Backward 
    optimizerG.zero_grad()
    generator.zero_grad()
    discriminator.zero_grad()
    generator_loss.backward()
    discriminator.zero_grad() #discriminator is not allowed to change generator weights
    optimizerG.step()
    # Discriminator : Forward   
    output = generator(input_image, target_image, groundtruth_image)
    relit_image, \
    input_light_latent, target_light_latent, groundtruth_light_latent, \
    input_scene_latent, target_scene_latent, groundtruth_scene_latent = output 
    disc_out_fake = discriminator(relit_image)
    disc_out_real = discriminator(target_image) # better than groundtruth_image ?
    discriminator_loss = gan_loss(disc_out_fake, disc_out_real)
    train_discriminator_loss += generator_loss.item()
    # Discriminator : Backward 
    optimizerD.zero_grad()
    generator.zero_grad()
    discriminator.zero_grad()
    discriminator_loss.backward()
    generator.zero_grad() #discriminator is not allowed to change generator weights
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
            writer.add_image('Visualization-Train/1-Input', make_grid(input_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Train/2-Input-Light-latent', make_grid(input_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Train/3-Target', make_grid(target_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Train/4-Target-Light-latent', make_grid(target_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Train/5-Ground-truth', make_grid(groundtruth_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Train/6-Ground-truth-Light-latent', make_grid(groundtruth_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Train/7-Relit', make_grid(relit_image[:SHOWN_SAMPLES]), step)
            writer.add_scalar('Loss/1-G-Loss-Train', train_generator_loss/TESTING_FREQ, step)
            writer.add_scalar('Loss/2-D-Loss-Train', train_discriminator_loss/TESTING_FREQ, step)
            writer.add_scalar('Score/1-Score-Train', train_score/TESTING_FREQ, step)
            print ('TrainLoss:', train_generator_loss/TESTING_FREQ, 'TrainScore:', train_score/TESTING_FREQ)
            # Reset train scalars
            train_generator_loss, train_discriminator_loss, train_score = 0., 0., 0.
            # Evaluate loop
            test_generator_loss, test_discriminator_loss, test_score = 0., 0., 0.
            testing_batches_counter = 0 
            while testing_batches_counter < TESTING_BATCHES :
                #Load batch
                try:
                    test_batch = next(test_dataloader_iter)
                except StopIteration:
                    test_dataloader_iter = iter(test_dataloader)
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
                generator_loss = reconstruction_loss(relit_image, groundtruth_image) + .3 * fool_gan_loss(disc_out_fake)
                discriminator_loss = gan_loss(disc_out_fake, disc_out_real)
                test_generator_loss += generator_loss.item()
                test_discriminator_loss += generator_loss.item()
                test_score += reconstruction_loss(input_image, groundtruth_image).item() / reconstruction_loss(relit_image, groundtruth_image).item()  
                # Update testing_batches_counter
                testing_batches_counter += 1            
            # Visualize test
            writer.add_image('Visualization-Test/1-Input', make_grid(input_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Test/2-Input-Light-latent', make_grid(input_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Test/3-Target', make_grid(target_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Test/4-Target-Light-latent', make_grid(target_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Test/5-Ground-truth', make_grid(groundtruth_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Test/6-Ground-truth-Light-latent', make_grid(groundtruth_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization-Test/7-Relit', make_grid(relit_image[:SHOWN_SAMPLES]), step)
            writer.add_scalar('Loss/2-G-Loss-Test', test_generator_loss/TESTING_BATCHES, step)
            writer.add_scalar('Loss/2-D-Loss-Test', test_discriminator_loss/TESTING_BATCHES, step)
            writer.add_scalar('Score/2-Score-Test', test_score/TESTING_BATCHES, step)
            print ('TestLoss:', test_generator_loss/TESTING_BATCHES, 'TestScore:', test_score/TESTING_BATCHES)
            # Switch to train mode
            generator.train()
            discriminator.train()  

# Store trained model
storage.save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
