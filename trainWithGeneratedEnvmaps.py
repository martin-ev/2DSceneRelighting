from torch import cat, randint, unique, FloatTensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Resize
from torchvision.utils import make_grid

from utils.metrics import psnr
from models.swapModels import GroundtruthEnvmapSwapNet
from models.loss import log_l2_loss
from numpy import sum
from tqdm import tqdm
from utils.dataset import InputTargetGroundtruthWithGeneratedEnvmapDataset, DifferentScene, DifferentLightDirection, \
    VALIDATION_DATA_PATH
from utils.storage import save_trained
from utils.device import setup_device
from utils import tensorboard


# Get used device
GPU_IDS = [3]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'generated_envmaps_6500_reconstruction_and_envmap_loss'
BATCH_SIZE = 25
NUM_WORKERS = 8
EPOCHS = 30
SIZE = 256
SAMPLED_TRAIN_SAMPLES = 50000
SAMPLED_TEST_SAMPLES = 5000

# Configure training objects
model = GroundtruthEnvmapSwapNet().to(device)
optimizer = Adam(model.parameters())

# Losses
reconstruction_loss = nn.L1Loss()
envmap_loss = log_l2_loss

# Configure data sets
transform = Resize(SIZE)
pairing_strategies = [DifferentScene(), DifferentLightDirection()]
train_dataset = InputTargetGroundtruthWithGeneratedEnvmapDataset(input_colors=[6500],
                                                                 target_colors=[6500],
                                                                 transform=transform,
                                                                 pairing_strategies=pairing_strategies)
test_dataset = InputTargetGroundtruthWithGeneratedEnvmapDataset(data_path=VALIDATION_DATA_PATH,
                                                                input_colors=[6500],
                                                                target_colors=[6500],
                                                                transform=transform,
                                                                pairing_strategies=pairing_strategies)

# Configure data loaders
# Sub-sampling:
# https://discuss.pytorch.org/t/train-on-a-fraction-of-the-data-set/16743/2
# https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/5
train_subset_indices = unique(randint(0, len(train_dataset), (SAMPLED_TRAIN_SAMPLES,)))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                              sampler=SubsetRandomSampler(train_subset_indices))
test_subset_indices = unique(randint(0, len(test_dataset), (SAMPLED_TEST_SAMPLES,)))
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                             sampler=SubsetRandomSampler(test_subset_indices))
TEST_BATCHES = len(test_dataloader)
TRAIN_SAMPLES = len(train_subset_indices)
TEST_SAMPLES = len(test_subset_indices)
print(f'Train dataset: {TRAIN_SAMPLES} samples, {len(train_dataloader)} batches.')
print(f'Test dataset: {TEST_SAMPLES} samples, {TEST_BATCHES} batches.')
print(f'Running with batch size: {BATCH_SIZE} for {EPOCHS} epochs.')


# Configure tensorboard
writer = tensorboard.setup_summary_writer(NAME)
tensorboard_process = tensorboard.start_tensorboard_process()
SHOWN_SAMPLES = 3
TRAIN_VISUALIZATION_FREQ = TRAIN_SAMPLES // BATCH_SIZE // 5
print(f'{SHOWN_SAMPLES} train samples will be visualized every {TRAIN_VISUALIZATION_FREQ} train batches.')


def normalize_image(latent):
    # See: https://discuss.pytorch.org/t/current-torch-min-does-not-support-multiple-dimensions/55577/2
    print(latent.size())
    x = latent.view(-1, 1536)
    x_min, x_max = x.min(dim=1)[0], x.max(dim=1)[0]
    return ((x - x_min) / (x_max - x_min)).view(-1, 3, 16, 32)


def visualize(in_img, out_img, gt_img, target_img,
              in_envmap, in_gt_envmap, target_envmap, target_gt_envmap,
              step, mode='Train'):
    writer.add_image(f'Visualization/{mode}/1-Input', make_grid(in_img[:SHOWN_SAMPLES]), step)
    writer.add_image(f'Visualization/{mode}/2-Relit', make_grid(out_img[:SHOWN_SAMPLES]), step)
    writer.add_image(f'Visualization/{mode}/3-Ground-truth', make_grid(gt_img[:SHOWN_SAMPLES]), step)
    writer.add_image(f'Visualization/{mode}/4-Target', make_grid(target_img[:SHOWN_SAMPLES]), step)

    input_envmaps = normalize_image(cat((in_envmap[:SHOWN_SAMPLES], in_gt_envmap[:SHOWN_SAMPLES]), dim=0))
    target_envmaps = normalize_image(cat((target_envmap[:SHOWN_SAMPLES], target_gt_envmap[:SHOWN_SAMPLES]), dim=0))
    writer.add_image(f'Env-map/{mode}/1-Input', make_grid(input_envmaps, nrow=SHOWN_SAMPLES), step)
    writer.add_image(f'Env-map/{mode}/2-Target', make_grid(target_envmaps, nrow=SHOWN_SAMPLES), step)


def report_loss(components, step, mode='Train'):
    total = FloatTensor(list(components.values())).sum()
    writer.add_scalar(f'Loss/{mode}/1-Total', total, step)
    writer.add_scalars(f'Loss/{mode}/2-Components', components, step)


def report_metrics(psnr_value, step, mode='Test'):
    writer.add_scalar(f'Metrics/{mode}/1-PSNR', psnr_value, step)


# Train loop
train_step = 0
for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    train_loss_reconstruction, train_loss_envmap = 0.0, 0.0
    for batch_idx, batch in enumerate(train_dataloader):
        x = batch[0][0]['image'].to(device)
        x_envmap = batch[0][1].to(device)
        target = batch[1][0]['image'].to(device)
        target_envmap = batch[1][1].to(device)
        groundtruth = batch[2]['image'].to(device)

        # Forward
        relit, pred_image_envmap, pred_target_envmap = model(x, target, groundtruth)
        loss_reconstruction = reconstruction_loss(relit, groundtruth)
        loss_image_envmap = envmap_loss(pred_image_envmap, x_envmap)
        loss_target_envmap = envmap_loss(pred_target_envmap, target_envmap)
        loss = loss_reconstruction + loss_image_envmap + loss_target_envmap

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss monitoring
        train_loss_reconstruction += loss_reconstruction.item()
        train_loss_envmap += loss_image_envmap.item() + loss_target_envmap.item()

        # Visualize current training progress
        if batch_idx % TRAIN_VISUALIZATION_FREQ == 0:
            visualize(x, relit, groundtruth, target,
                      pred_image_envmap, x_envmap, pred_target_envmap, target_envmap,
                      train_step, 'Train')
            report_loss({
                'Reconstruction': train_loss_reconstruction,
                'Envmap': train_loss_envmap,
            }, train_step, 'Train')

            train_loss_reconstruction, train_loss_envmap = 0.0, 0.0
            train_step += 1

    # Clean up memory (see: https://repl.it/@nickangtc/python-del-multiple-variables-in-single-line)
    del x, x_envmap, target, target_envmap, groundtruth, relit, pred_image_envmap, pred_target_envmap

    # Evaluate
    model.eval()
    test_loss_reconstruction = 0.0
    test_loss_image_envmap, test_loss_target_envmap = 0.0, 0.0
    test_psnr = 0.0
    random_batch_id = randint(0, TEST_BATCHES, (1,))
    for test_batch_idx, test_batch in enumerate(test_dataloader):
        test_x = batch[0][0]['image'].to(device)
        test_x_envmap = batch[0][1].to(device)
        test_target = batch[1][0]['image'].to(device)
        test_target_envmap = batch[1][1].to(device)
        test_groundtruth = batch[2]['image'].to(device)

        # Inference
        test_relit, test_pred_image_envmap, test_pred_target_envmap = model(test_x, test_target, test_groundtruth)

        # Test loss
        test_loss_reconstruction += reconstruction_loss(test_relit, test_groundtruth).item()
        test_loss_image_envmap += envmap_loss(test_pred_image_envmap, test_x_envmap).item()
        test_loss_target_envmap += envmap_loss(test_pred_target_envmap, test_target_envmap)
        test_psnr += psnr(test_relit, test_groundtruth)

        # Visualize random evaluation batch
        if test_batch_idx == random_batch_id:
            visualize(test_x, test_relit, test_groundtruth, test_target,
                      test_pred_image_envmap, test_x_envmap, test_pred_target_envmap, test_target_envmap,
                      epoch, 'Test')

    # Clean up memory
    del test_x, test_x_envmap, test_target, test_target_envmap, test_groundtruth, test_relit, test_pred_image_envmap,\
        test_pred_target_envmap

    # Report test metrics
    report_loss({
        '1-Reconstruction': test_loss_reconstruction,
        '2-Image-env-map': test_loss_image_envmap,
        '3-Target-env-map': test_loss_target_envmap
    }, epoch, 'Test')
    report_metrics(test_psnr / TEST_SAMPLES, epoch, 'Test')

# Store trained model
save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
