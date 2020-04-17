import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.utils import make_grid

from models.illumination_swap import IlluminationSwapNet
from models.loss import log_l2_loss
from utils import dataset, storage, tensorboard
from utils.device import setup_device


# Get used device
GPU_IDS = [2]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'illumination_swap_only_abandonned_6500'
BATCH_SIZE = 5
NUM_WORKERS = 4
EPOCHS = 50
VISUALIZATION_FREQ = 100  # every how many batches tensorboard is updated with new images

# Configure training objects
model = IlluminationSwapNet().to(device)
optimizer = Adam(model.parameters())

# Losses
reconstruction_loss = nn.L1Loss()
env_map_loss = log_l2_loss

# Configure dataloader
dataset = dataset.DifferentTargetSceneDataset(locations=['scene_abandonned_city_54'],
                                              input_colors=['6500'],
                                              target_colors=['6500'],
                                              transform=Resize(256))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Configure tensorboard
writer = tensorboard.setup_summary_writer(NAME)
tensorboard_process = tensorboard.start_tensorboard_process()

# writer.add_text('Visualization/Images', 'Following rows are: input, relighted image, ground-truth and target')
# writer.add_text('Visualization/Env-map', 'Env-maps for relighted image (top) and ground-truth (bottom)')

# Train loop
for epoch in range(1, EPOCHS+1):
    train_loss, train_loss_reconstruction, train_loss_env_map = 0, 0, 0
    for batch_idx, batch in enumerate(dataloader):
        x = batch[0][0]['image'].to(device)
        target = batch[0][1]['image'].to(device)
        ground_truth = batch[1]['image'].to(device)

        # Forward
        model.train()
        relighted_image, relighted_env_map, gt_env_map = model(x, target, ground_truth)
        loss1 = reconstruction_loss(relighted_image, ground_truth)
        loss2 = env_map_loss(relighted_env_map, gt_env_map)
        loss = loss1 + loss2

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_reconstruction += loss1.item()
        train_loss_env_map += loss2.item()

        # Visualize current progress
        if batch_idx % VISUALIZATION_FREQ == 0:
            writer.add_image('Visualization/Input', make_grid(x, nrow=BATCH_SIZE), epoch)
            writer.add_image('Visualization/Relighted', make_grid(relighted_image, nrow=BATCH_SIZE), epoch)
            writer.add_image('Visualization/Ground-truth', make_grid(ground_truth, nrow=BATCH_SIZE), epoch)
            writer.add_image('Visualization/Target', make_grid(target,nrow=BATCH_SIZE), epoch)

            writer.add_image('Env-map/Relighted', make_grid(relighted_env_map.view(-1, 3, 16, 32), nrow=BATCH_SIZE), epoch)
            writer.add_image('Env-map/Ground-truth', make_grid(gt_env_map.view(-1, 3, 16, 32), nrow=BATCH_SIZE), epoch)

    # Evaluate
    model.eval()
    # TODO: Add test set evaluation here

    # Update tensorboard training losses
    writer.add_scalar('Loss/Total training loss', train_loss, epoch)
    writer.add_scalars('Loss/Training loss components', {
        'Reconstruction': train_loss_reconstruction,
        'Env map': train_loss_env_map
    }, epoch)

# Store trained model
storage.save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
