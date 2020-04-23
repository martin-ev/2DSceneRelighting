import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.utils import make_grid

from models.illumination_swap import IlluminationSwapNet
from models.loss import log_l2_loss
from tqdm import tqdm
from utils.dataset import InputTargetGroundtruthDataset, DifferentScene, DifferentLightDirection
from utils.storage import save_trained
from utils.tensorboard import start_tensorboard_process, stop_tensorboard_process, setup_summary_writer
from utils.device import setup_device


# Get used device
GPU_IDS = [3]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'illumination_swap_only_abandonned_6500'
BATCH_SIZE = 25
NUM_WORKERS = 8
EPOCHS = 30
SIZE = 256

# Configure training objects
model = IlluminationSwapNet().to(device)
optimizer = Adam(model.parameters())

# Losses
reconstruction_loss = nn.L1Loss()
env_map_loss = log_l2_loss

# Configure dataloader
dataset = InputTargetGroundtruthDataset(locations=['scene_abandonned_city_54'],
                                        input_colors=['6500'],
                                        target_colors=['6500'],
                                        transform=Resize(SIZE),
                                        pairing_strategies=[DifferentScene(), DifferentLightDirection()])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
DATASET_SIZE = len(dataset)
print(f'Dataset contains {DATASET_SIZE} samples.')
print(f'Running with batch size: {BATCH_SIZE} for {EPOCHS} epochs.')

# Configure tensorboard
writer = setup_summary_writer(NAME)
tensorboard_process = start_tensorboard_process()
SHOWN_SAMPLES = 3
VISUALIZATION_FREQ = DATASET_SIZE // BATCH_SIZE // 10  # every how many batches tensorboard is updated with new images
print(f'{SHOWN_SAMPLES} samples will be visualized every {VISUALIZATION_FREQ} batches.')

# Train loop
for epoch in range(1, EPOCHS+1):
    train_loss, train_loss_reconstruction, train_loss_env_map = 0, 0, 0
    print(f'Epoch {epoch}:')
    for batch_idx, batch in tqdm(enumerate(dataloader)):
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
            writer.add_image('Visualization/1-Input', make_grid(x[:SHOWN_SAMPLES]), epoch)
            writer.add_image('Visualization/2-Relighted', make_grid(relighted_image[:SHOWN_SAMPLES]), epoch)
            writer.add_image('Visualization/3-Ground-truth', make_grid(ground_truth[:SHOWN_SAMPLES]), epoch)
            writer.add_image('Visualization/4-Target', make_grid(target[:SHOWN_SAMPLES]), epoch)

            writer.add_image('Env-map/1-Relighted', make_grid(relighted_env_map[:SHOWN_SAMPLES].view(-1, 3, 16, 32)),
                             epoch)
            writer.add_image('Env-map/2-Ground-truth', make_grid(gt_env_map[:SHOWN_SAMPLES].view(-1, 3, 16, 32)), epoch)

    # Evaluate
    model.eval()
    # TODO: Add test set evaluation here

    # Update tensorboard training losses
    writer.add_scalar('Loss/1-Total', train_loss, epoch)
    writer.add_scalars('Loss/2-Components', {
        '1-Reconstruction': train_loss_reconstruction,
        '2-Env-map': train_loss_env_map
    }, epoch)

# Store trained model
save_trained(model, NAME)

# Terminate tensorboard
stop_tensorboard_process(tensorboard_process)
