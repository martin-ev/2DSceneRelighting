import torch.nn as nn
import argparse

from torch import cat, randint, unique, FloatTensor, no_grad
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from lpips_pytorch import LPIPS
from kornia.color import hsv_to_rgb

from utils.metrics import psnr
from models.swapModels import SinglePortraitEnvmapSwapNet, \
    SinglePortraitEnvmapNetSplitter, SinglePortraitEnvmapNetAssembler, \
    SceneEnvmapNetSplitter, SceneEnvmapNetAssembler
from utils.losses import log_l2_loss, hsv_envmap_loss
from utils.dataset import InputTargetGroundtruthWithGeneratedEnvmapDataset, DifferentScene, DifferentLightDirection, \
    VALIDATION_DATA_PATH
from utils.storage import save_trained, save_checkpoint
from utils.device import setup_device
from utils import tensorboard


# Get used device
GPU_IDS = [3]
device = setup_device(GPU_IDS)

# Parameters
NAME = 'generated_envmaps_scene_light_split'
BATCH_SIZE = 25
NUM_WORKERS = 6
EPOCHS = 20
SIZE = 256
SAMPLED_TRAIN_SAMPLES = 400000
SAMPLED_TEST_SAMPLES = 10000

# Arguments
# List arguments parsing: https://stackoverflow.com/a/15753721
parser = argparse.ArgumentParser(description='Illumination Swap network with configurable skip connections')
parser.add_argument('-d', '--disabled-skip-connections',
                    dest='disabled_skip_connections',
                    nargs='*',
                    type=int,
                    help='Numbers of encoder layers that will not be propagated to the decoder as skip connections')
parser.add_argument('-a', '--add-target-skip-connections',
                    dest='target_skip_connections',
                    nargs='*',
                    type=int,
                    help='Numbers of encoder layers from target image pass that will replace original skip '
                         'connections in the decoder. Is overridden by --disabled-skip-connection, i.e. if skip '
                         'connection from particular layer is disabled also target skip connection from this layer '
                         'will not be used.')
parser.add_argument('--latent',
                    dest='latent',
                    choices=['light', 'scene-light'],
                    required=True,
                    help='Specifies what information should be encoded in the latent representation. "light" - latent '
                         'represents only predicted envmap of the input image. "scene_light" - latent encodes both '
                         'the predicted envmap of the input image and scene (content) information.')
parser.add_argument('--loss',
                    dest='loss',
                    choices=['l1', 'lpips'],
                    required=True,
                    help='Loss to be used as reconstruction loss. Can be either MAE loss ("l1") or LPIPS ("lpips").')
ARGUMENTS = parser.parse_args()


# Configure training objects
splitter, assembler, envmap_colorspace = None, None, None
if ARGUMENTS.latent == 'light':
    splitter = SinglePortraitEnvmapNetSplitter()
    assembler = SinglePortraitEnvmapNetAssembler()
    envmap_colorspace = 'rgb'
elif ARGUMENTS.latent == 'scene-light':
    splitter = SceneEnvmapNetSplitter(scene_latent_channels=1024)
    assembler = SceneEnvmapNetAssembler(scene_latent_channels=1024)
    envmap_colorspace = 'hsv'

model = SinglePortraitEnvmapSwapNet(
    splitter=splitter,
    assembler=assembler,
    disabled_skip_connections_ids=ARGUMENTS.disabled_skip_connections,
    target_skip_connections_ids=ARGUMENTS.target_skip_connections
).to(device)
optimizer = Adam(model.parameters())
print('Model:', model.__class__.__name__)
print('Disabled skip connections:', ARGUMENTS.disabled_skip_connections)
print('Target skip connections:', ARGUMENTS.target_skip_connections)

# Losses
reconstruction_loss = None
if ARGUMENTS.loss == 'l1':
    reconstruction_loss = nn.L1Loss().to(device)
elif ARGUMENTS.loss == 'lpips':
    reconstruction_loss = LPIPS(device).to(device)

envmap_loss = None
if envmap_colorspace == 'rgb':
    envmap_loss = log_l2_loss
else:
    envmap_loss = hsv_envmap_loss

# Configure data sets
transform = Resize(SIZE)
pairing_strategies = [DifferentScene(), DifferentLightDirection()]
train_dataset = InputTargetGroundtruthWithGeneratedEnvmapDataset(transform=transform,
                                                                 pairing_strategies=pairing_strategies,
                                                                 mode=envmap_colorspace)
test_dataset = InputTargetGroundtruthWithGeneratedEnvmapDataset(data_path=VALIDATION_DATA_PATH,
                                                                transform=transform,
                                                                pairing_strategies=pairing_strategies,
                                                                mode=envmap_colorspace)

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
TRAIN_VISUALIZATION_FREQ = TRAIN_SAMPLES // BATCH_SIZE // 4
CHECKPOINT_EVERY = 2  # save model checkpoint every n epochs
print(f'{SHOWN_SAMPLES} train samples will be visualized every {TRAIN_VISUALIZATION_FREQ} train batches.')


def normalize_image(light_latent):
    # See: https://discuss.pytorch.org/t/current-torch-min-does-not-support-multiple-dimensions/55577/2
    if ARGUMENTS.latent == 'scene-light':
        envmap_hs, envmap_v = light_latent.split([2, 512], dim=1)
        envmap_hsv = cat((envmap_hs.repeat_interleave(512, dim=1), envmap_v), dim=1).view(-1, 3, 16, 32)
        return hsv_to_rgb(envmap_hsv / 255.)
    elif ARGUMENTS.latent == 'light':
        # TODO: it should not be normalized across channels
        envmap = light_latent.view(-1, 1536)
        envmap_min = envmap.min(dim=1)[0].unsqueeze(1).expand(-1, 1536)
        envmap_max = envmap.max(dim=1)[0].unsqueeze(1).expand(-1, 1536)
        return ((envmap - envmap_min) / (envmap_max - envmap_min)).view(-1, 3, 16, 32)


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
    n_batches_since_visualization = 0
    for batch_idx, batch in enumerate(train_dataloader):
        n_batches_since_visualization += 1

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
                'Reconstruction': train_loss_reconstruction / n_batches_since_visualization,
                'Envmap': train_loss_envmap / n_batches_since_visualization,
            }, train_step, 'Train')

            train_loss_reconstruction, train_loss_envmap = 0.0, 0.0
            n_batches_since_visualization = 0
            train_step += 1

    # Clean up memory (see: https://repl.it/@nickangtc/python-del-multiple-variables-in-single-line)
    del x, x_envmap, target, target_envmap, groundtruth, relit, pred_image_envmap, pred_target_envmap

    # Saving checkpoint
    if epoch % CHECKPOINT_EVERY == 0:
        save_checkpoint(model.state_dict(), optimizer.state_dict(), NAME + '_' + str(epoch))

    # Evaluate
    with no_grad():
        model.eval()
        test_loss_reconstruction = 0.0
        test_loss_image_envmap, test_loss_target_envmap = 0.0, 0.0
        test_psnr = 0.0
        random_batch_id = randint(0, TEST_BATCHES, (1,))
        for test_batch_idx, test_batch in enumerate(test_dataloader):
            test_x = test_batch[0][0]['image'].to(device)
            test_x_envmap = test_batch[0][1].to(device)
            test_target = test_batch[1][0]['image'].to(device)
            test_target_envmap = test_batch[1][1].to(device)
            test_groundtruth = test_batch[2]['image'].to(device)

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
        del test_x, test_x_envmap, test_target, test_target_envmap, test_groundtruth, test_relit, \
            test_pred_image_envmap, test_pred_target_envmap

        # Report test metrics
        report_loss({
            '1-Reconstruction': test_loss_reconstruction / TEST_BATCHES,
            '2-Image-env-map': test_loss_image_envmap / TEST_BATCHES,
            '3-Target-env-map': test_loss_target_envmap / TEST_BATCHES
        }, epoch, 'Test')
        report_metrics(test_psnr / TEST_BATCHES, epoch, 'Test')

# Store trained model
save_trained(model, NAME)

# Terminate tensorboard
tensorboard.stop_tensorboard_process(tensorboard_process)
