import json
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import no_grad, randperm
from torch.nn.functional import mse_loss
from tqdm import tqdm

# imports used for creating models based on their class name
from models.swapModels import AnOtherSwapNet, SinglePortraitEnvmapSwapNet, \
    SinglePortraitEnvmapNetSplitter, SinglePortraitEnvmapNetAssembler, SceneEnvmapNetSplitter, SceneEnvmapNetAssembler
from utils.dataset import InputTargetGroundtruthDataset, DifferentScene, DifferentLightDirection, VALIDATION_DATA_PATH
from utils.device import setup_device
from utils.storage import load_checkpoint, load_trained
from utils.metrics import ssim as compute_ssim
from utils.metrics import psnr as compute_psnr
from lpips_pytorch import LPIPS


RESULTS_FILE = '/ivrldata1/students/team6/results.json'
USED_TEST_SAMPLES = 10000


# Get used device
GPU_IDS = [3]
device = setup_device(GPU_IDS)

# LPIPS setup
compute_lpips = LPIPS(device).to(device)

# Model definitions
models = {
    'illumination_predicter': {
        'path': 'generatorAnOtherSwapNetIlluminationPerdicterNoGAN1590479412.3209212',
        'class': 'AnOtherSwapNet',
        'stored_model_only': True,
        'parametrized': False
    },
    'envmap': {
        'path': 'generated_envmaps_all_proper_color_scaling_8',
        'class': 'SinglePortraitEnvmapSwapNet',
        'splitter_class': 'SinglePortraitEnvmapNetSplitter',
        'assembler_class': 'SinglePortraitEnvmapNetAssembler',
        'stored_model_only': False,
        'parametrized': True
    },
    'envmap_with_scene': {
        'path': 'generated_envmaps_scene_light_split_6',  # TODO: replace the epoch
        'class': 'SinglePortraitEnvmapSwapNet',
        'splitter_class': 'SceneEnvmapNetSplitter',
        'assembler_class': 'SceneEnvmapNetAssembler',
        'stored_model_only': False,
        'parametrized': True
    }
}

# Dataset & dataloader
pairing_strategies = [DifferentScene(), DifferentLightDirection()]
test_dataset = InputTargetGroundtruthDataset(data_path=VALIDATION_DATA_PATH,
                                             transform=Resize(256),
                                             pairing_strategies=pairing_strategies)
indices = randperm(len(test_dataset))[:USED_TEST_SAMPLES]
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=2, sampler=SubsetRandomSampler(indices))
TEST_SET_SIZE = len(test_dataloader)


def load_model(configuration):
    if configuration['stored_model_only']:
        net = eval(configuration['class'])()
        return load_trained(net, configuration['path'])
    elif configuration['parametrized']:
        splitter = eval(configuration['splitter_class'])(scene_latent_channels=1024)
        assembler = eval(configuration['assembler_class'])(scene_latent_channels=1024)
        net = eval(configuration['class'])(splitter, assembler)
        checkpoint = load_checkpoint(configuration['path'])
        net.load_state_dict(checkpoint['model_state_dict'])
        return net


# Evaluate
print('Computing metrics on', TEST_SET_SIZE, 'test samples')
results = {}
with no_grad():
    for model_name, config in models.items():
        print('Computing metrics for', model_name)

        # Load model
        model = load_model(config).to(device)
        model.eval()

        # Prepare metrics
        mse, ssim, psnr, lpips = 0., 0., 0., 0.

        # Compute metrics on test set
        for batch in tqdm(test_dataloader):
            image = batch[0][0]['image'].to(device)
            target = batch[0][1]['image'].to(device)
            groundtruth = batch[1]['image'].to(device)

            output = model(image, target, groundtruth)
            relit = output[0]

            # Compute metrics on sample
            mse += mse_loss(relit, groundtruth)
            ssim += compute_ssim(relit, groundtruth)
            psnr += compute_psnr(relit, groundtruth)
            lpips += compute_lpips(relit, groundtruth)

        # Average over test set
        mse /= TEST_SET_SIZE
        ssim /= TEST_SET_SIZE
        psnr /= TEST_SET_SIZE
        lpips /= TEST_SET_SIZE

        # Record
        results[model_name] = {
            'mse': mse,
            'ssim': ssim,
            'psnr': psnr,
            'lpips': lpips
        }

# Store the results, see https://stackoverflow.com/a/26057360
print('Done, saving results')
with open(RESULTS_FILE, 'w') as file:
    json.dump(results, file)
