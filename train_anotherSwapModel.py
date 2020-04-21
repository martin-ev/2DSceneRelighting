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

from utils.dataset import DifferentTargetSceneDataset
from torch.utils.data import DataLoader

from models.anOtherSwapNetSmaller import SwapModel


# Get used device
GPU_IDS = [1]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'anOtherSwapNet'
BATCH_SIZE = 50
NUM_WORKERS = 8
EPOCHS = 30
SIZE = 256

# Configure training objects
model = SwapModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0)

# Losses
distance = nn.MSELoss().to(device)

# Configure dataloader
train_dataset = DifferentTargetSceneDataset(locations=['scene_abandonned_city_54'],
                                              transform=transforms.Resize(SIZE))
train_dataloader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
DATASET_SIZE = len(train_dataset)
print(f'Dataset contains {DATASET_SIZE} samples.')
print(f'Running with batch size: {BATCH_SIZE} for {EPOCHS} epochs.')

# Configure tensorboard
writer = tensorboard.setup_summary_writer(NAME)
tensorboard_process = tensorboard.start_tensorboard_process()
SHOWN_SAMPLES = 3
VISUALIZATION_FREQ = 100 #DATASET_SIZE // BATCH_SIZE // 10  # every how many batches tensorboard is updated with new images
print(f'{SHOWN_SAMPLES} samples will be visualized every {VISUALIZATION_FREQ} batches.')

# Train loop
for epoch in range(1, EPOCHS+1):
    train_loss, train_score = 0., 0.
    print(f'Epoch {epoch}:')
    
    step = 0
    sub_train_loss, sub_train_score = 0., 0.
        
    for batch_idx, batch in enumerate(train_dataloader):
        
        input_image = batch[0][0]['image'].to(device)
        target_image = batch[0][1]['image'].to(device)
        groundtruth_image = batch[1]['image'].to(device)
        input_color = batch[0][0]['color'].to(device)
        target_color = batch[0][1]['color'].to(device)
        input_direction = batch[0][0]['direction'].to(device)
        target_direction = batch[0][1]['direction'].to(device)
        
        # Forward
        model.train()     
        
        output = model(input_image, target_image, groundtruth_image, encode_pred = True)
        
        groundtruth_scene_latent, input_scene_latent, target_scene_latent, relighted_scene_latent, \
        groundtruth_light_latent, input_light_latent, target_light_latent, relighted_light_latent, \
        groundtruth_light_predic, input_light_predic, target_light_predic, \
        relighted_image, relighted_image2 = output  
        
        loss = 1. * distance(relighted_image, groundtruth_image) 
                

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_score += distance(input_image, groundtruth_image).item() / loss.item()
        sub_train_loss += loss.item()
        sub_train_score += distance(input_image, groundtruth_image).item() / loss.item()
        
        # Visualize current progress
        if (1+batch_idx) % VISUALIZATION_FREQ == 0:
            writer.add_image('Visualization/1-Input', make_grid(input_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization/2-Target', make_grid(target_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization/3-Ground-truth', make_grid(groundtruth_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization/4-Relighted', make_grid(relighted_image[:SHOWN_SAMPLES]), step)
            writer.add_image('Visualization/5-Relighted2', make_grid(relighted_image2[:SHOWN_SAMPLES]), step)

            writer.add_image('Light-latent/1-Input', make_grid(input_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Light-latent/2-Target', make_grid(target_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Light-latent/3-Ground-truth', make_grid(groundtruth_light_latent[:SHOWN_SAMPLES]), step)
            writer.add_image('Light-latent/4-Relighted', make_grid(relighted_light_latent[:SHOWN_SAMPLES]), step)
            
            step += 1
            writer.add_scalar('Sub-loss/1-Loss', sub_train_loss, step)
            writer.add_scalar('Sub-score/1-Score', sub_train_score, step)
            print ('Sub-loss/1-Loss:', sub_train_loss, 'Sub-score/1-Score:', sub_train_score)
            sub_train_loss = 0.
            sub_train_score = 0.
        
       
    # Evaluate
    model.eval()
    # TODO: Add test set evaluation here

    # Update tensorboard training losses
    writer.add_scalar('Loss/1-Loss', train_loss, epoch)
    writer.add_scalar('Score/1-Score', train_loss, epoch)

# Store trained model
storage.save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
