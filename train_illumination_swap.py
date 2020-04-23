from torch import randint
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.utils import make_grid

from utils.metrics import psnr
from models.illumination_swap import IlluminationSwapNet
from models.loss import log_l2_loss
from tqdm import tqdm
from utils.dataset import InputTargetGroundtruthDataset, DifferentScene, DifferentLightDirection
from utils.storage import save_trained
from utils.device import setup_device
from utils import tensorboard


# Get used device
GPU_IDS = [2]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'illumination_swap_all_reconstruction_and_envmap_loss'
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


def compute_losses(relighted, image_env_map, target_env_map, ground_truth_env_map):
    reconstruct_loss = reconstruction_loss(relighted, ground_truth)
    envmap_loss = env_map_loss(target_env_map, ground_truth_env_map) / env_map_loss(image_env_map, ground_truth_env_map)
    return reconstruct_loss, envmap_loss


# Configure dataloader
transform = Resize(SIZE)
pairing_strategies = [DifferentScene(), DifferentLightDirection()]
train_dataset = InputTargetGroundtruthDataset(transform=transform,
                                              pairing_strategies=pairing_strategies)
test_dataset = InputTargetGroundtruthDataset(transform=transform,
                                             pairing_strategies=pairing_strategies)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
TRAIN_DATASET_SIZE = len(train_dataset)
TEST_DATASET_SIZE = len(test_dataset)
print(f'Train dataset: {TRAIN_DATASET_SIZE} samples.')
print(f'Test dataset: {TEST_DATASET_SIZE} samples.')
print(f'Running with batch size: {BATCH_SIZE} for {EPOCHS} epochs.')


def unpack_batch(data_batch):
    inp = data_batch[0][0]['image'].to(device)
    tar = data_batch[0][1]['image'].to(device)
    gt = data_batch[1]['image'].to(device)
    return inp, tar, gt


# Configure tensorboard
writer = tensorboard.setup_summary_writer(NAME)
tensorboard_process = tensorboard.start_tensorboard_process()
SHOWN_SAMPLES = 3
TRAIN_VISUALIZATION_FREQ = TRAIN_DATASET_SIZE // BATCH_SIZE // 10
print(f'{SHOWN_SAMPLES} samples will be visualized every {TRAIN_VISUALIZATION_FREQ} train batches.')


def visualize(in_img, out_img, gt_img, target_img, in_envmap, gt_envmap, target_envmap, step, mode='Train'):
    writer.add_image(f'Visualization/{mode}/1-Input', make_grid(in_img[:SHOWN_SAMPLES]), step)
    writer.add_image(f'Visualization/{mode}/2-Relighted', make_grid(out_img[:SHOWN_SAMPLES]), step)
    writer.add_image(f'Visualization/{mode}/3-Ground-truth', make_grid(gt_img[:SHOWN_SAMPLES]), step)
    writer.add_image(f'Visualization/{mode}/4-Target', make_grid(target_img[:SHOWN_SAMPLES]), step)

    writer.add_image(f'Env-map/{mode}/1-Input', make_grid(in_envmap[:SHOWN_SAMPLES].view(-1, 3, 16, 32)), step)
    writer.add_image(f'Env-map/{mode}/2-Ground-truth', make_grid(gt_envmap[:SHOWN_SAMPLES].view(-1, 3, 16, 32)), step)
    writer.add_image(f'Env-map/{mode}/3-Target', make_grid(target_envmap[:SHOWN_SAMPLES].view(-1, 3, 16, 32)), step)


def report_loss(component_reconstruction, component_envmap, step, mode='Train'):
    writer.add_scalar(f'Loss/{mode}/1-Total', component_reconstruction + component_envmap, step)
    writer.add_scalars(f'Loss/{mode}/2-Components', {
        '1-Reconstruction': component_reconstruction,
        '2-Env-map': component_envmap
    }, step)


def report_metrics(psnr_value, step, mode='Train'):
    writer.add_scalar(f'Metrics/{mode}/1-PSNR', psnr_value, step)


# Train loop
for epoch in range(1, EPOCHS+1):
    print(f'Epoch {epoch}:')

    # Train
    model.train()
    train_loss_reconstruction, train_loss_env_map = 0, 0
    train_psnr = 0.0
    for batch_idx, batch in tqdm(enumerate(train_dataloader)):
        x, target, ground_truth = unpack_batch(batch)

        # Forward
        relighted_image, image_em, target_em, ground_truth_em = model(x, target, ground_truth)
        loss1, loss2 = compute_losses(relighted_image, image_em, target_em, ground_truth_em)
        loss = loss1 + loss2
        train_psnr += psnr(relighted_image, ground_truth)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_reconstruction += loss1.item()
        train_loss_env_map += loss2.item()

        # Visualize current progress
        if batch_idx % TRAIN_VISUALIZATION_FREQ == 0:
            visualize(x, relighted_image, ground_truth, target,
                      image_em, ground_truth_em, target_em,
                      epoch, 'Train')

    # Evaluate
    model.eval()
    test_loss_reconstruction, test_loss_env_map = 0, 0
    test_psnr = 0.0
    random_batch_id = randint(0, TEST_DATASET_SIZE, (1,))
    for test_batch_idx, test_batch in enumerate(test_dataloader):
        test_x, test_target, test_ground_truth = unpack_batch(test_batch)

        # Inference
        test_relighted_image, test_image_em, test_target_em, test_ground_truth_em = \
            model(test_x, test_target, test_ground_truth)

        # Test loss
        loss1, loss2 = compute_losses(test_relighted_image, test_image_em, test_target_em, test_ground_truth_em)
        test_loss_reconstruction += loss1.item()
        test_loss_env_map += loss2.item()
        test_psnr = psnr(test_relighted_image, test_ground_truth)

        # Visualize randomly selected batch
        if test_batch_idx == random_batch_id:
            visualize(test_x, test_relighted_image, test_ground_truth, test_target,
                      test_image_em, test_ground_truth_em, test_target_em,
                      epoch, 'Test')

    # Update tensorboard with losses and metrics
    train_psnr /= TRAIN_DATASET_SIZE
    test_psnr /= TEST_DATASET_SIZE
    report_loss(train_loss_reconstruction, train_loss_env_map, epoch, 'Train')
    report_loss(test_loss_reconstruction, test_loss_env_map, epoch, 'Test')
    report_metrics(train_psnr, epoch, 'Train')
    report_metrics(test_psnr, epoch, 'Test')

# Store trained model
save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
