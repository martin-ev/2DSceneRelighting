import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid

from utils.storage import save_trained, load_trained
from utils.device import setup_device
import utils.tensorboard as tensorboard

from utils.dataset import DifferentTargetSceneDataset, TRAIN_DATA_PATH, VALIDATION_DATA_PATH
from torch.utils.data import DataLoader

from models.anOtherSwapNetSmaller import SwapModel


# Get used device
GPU_IDS = [0]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'anOtherSwapNet-allLocations-evaluation'
BATCH_SIZE = 25
NUM_WORKERS = 8
SIZE = 256
TRAIN_DURATION = 60000

# Configure training objects
model = SwapModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0)

# Losses
distance = nn.MSELoss().to(device)

# Configure dataloader
train_dataset = DifferentTargetSceneDataset(transform=transforms.Resize(SIZE), data_path=TRAIN_DATA_PATH)
test_dataset = DifferentTargetSceneDataset(transform=transforms.Resize(SIZE), data_path=VALIDATION_DATA_PATH)
train_dataloader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
DATASET_SIZE = len(train_dataset)
print(f'Dataset contains {DATASET_SIZE} samples.')
print(f'Running with batch size: {BATCH_SIZE} for {TRAIN_DURATION} batches.')

# Configure tensorboard
writer = tensorboard.setup_summary_writer(NAME)
tensorboard_process = tensorboard.start_tensorboard_process()
SHOWN_SAMPLES = 3
TESTING_FREQ = 100 # every how many batches model is tested and tensorboard is updated
TESTING_BATCHES = 100 # how many batches for testing
print(f'{SHOWN_SAMPLES} samples will be visualized every {TESTING_FREQ} batches.')

# Train loop
step = 0 #1 step is TESTING_FREQ batches
train_loss, train_score = 0., 0.

test_dataloader_iter = iter(test_dataloader)
train_dataloader_iter = iter(train_dataloader)

model.train()  
train_batches_counter = 0 
while train_batches_counter < TRAIN_DURATION :
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

    # Forward   

    output = model(input_image, target_image, groundtruth_image, encode_pred = True)

    groundtruth_scene_latent, input_scene_latent, target_scene_latent, relighted_scene_latent, \
    groundtruth_light_latent, input_light_latent, target_light_latent, relighted_light_latent, \
    groundtruth_light_predic, input_light_predic, target_light_predic, \
    relighted_image, relighted_image2 = output  

    loss = 1. * distance(relighted_image, groundtruth_image)
    train_loss += loss.item()
    train_score += distance(input_image, groundtruth_image).item() / loss.item()

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_batches_counter += 1

    # Visualize current progress
    step, modulo = divmod(train_batches_counter, TESTING_FREQ)
    if modulo == 0:
        writer.add_image('Visualization-Train/1-Input', make_grid(input_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Train/2-Input-Light-latent', make_grid(input_light_latent[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Train/3-Target', make_grid(target_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Train/4-Target-Light-latent', make_grid(target_light_latent[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Train/5-Ground-truth', make_grid(groundtruth_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Train/6-Ground-truth-Light-latent', make_grid(groundtruth_light_latent[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Train/7-Relighted', make_grid(relighted_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Train/8-Relighted-Light-latent', make_grid(relighted_light_latent[:SHOWN_SAMPLES]), step)

        writer.add_scalar('Loss/1-Loss-Train', train_loss, step)
        writer.add_scalar('Score/1-Score-Train', train_score, step)
        print ('Sub-loss/1-Loss:', train_loss, 'Sub-score/1-Score:', train_score)
        train_loss = 0.
        train_score = 0.

        # Evaluate
        model.eval()
        test_loss = 0.
        test_score = 0.
        testing_batches_counter = 0 
        while testing_batches_counter < TESTING_BATCHES :
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

            output = model(input_image, target_image, groundtruth_image, encode_pred = True)

            groundtruth_scene_latent, input_scene_latent, target_scene_latent, relighted_scene_latent, \
            groundtruth_light_latent, input_light_latent, target_light_latent, relighted_light_latent, \
            groundtruth_light_predic, input_light_predic, target_light_predic, \
            relighted_image, relighted_image2 = output  

            loss = 1. * distance(relighted_image, groundtruth_image)
            test_loss += loss.item()
            test_score += distance(input_image, groundtruth_image).item() / loss.item()
                
            testing_batches_counter += 1
            
        writer.add_image('Visualization-Test/1-Input', make_grid(input_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Test/2-Input-Light-latent', make_grid(input_light_latent[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Test/3-Target', make_grid(target_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Test/4-Target-Light-latent', make_grid(target_light_latent[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Test/5-Ground-truth', make_grid(groundtruth_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Test/6-Ground-truth-Light-latent', make_grid(groundtruth_light_latent[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Test/7-Relighted', make_grid(relighted_image[:SHOWN_SAMPLES]), step)
        writer.add_image('Visualization-Test/8-Relighted-Light-latent', make_grid(relighted_light_latent[:SHOWN_SAMPLES]), step)

        writer.add_scalar('Loss/2-Loss-Test', test_loss, step)
        writer.add_scalar('Score/2-Score-Test', test_score, step)
        print ('Sub-loss/1-Loss:', test_loss, 'Sub-score/1-Score:', test_score)
           
        model.train() 
        
             
#Store trained model
storage.save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
