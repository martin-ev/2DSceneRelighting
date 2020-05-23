from collections import OrderedDict
from torch import device, cuda
import os

import torch

def setup_device(gpu_ids):
    """
    Creates a torch.device
    @param gpu_ids: ID(s) (list of integers) of the GPU(s) that should be used if they're available
    @return: device which can be used e.g. for torch.Tensor.to(<device>)
    """
    gpu_ids = [str(gpu_id) for gpu_id in sorted(gpu_ids) if 0 <= gpu_id <= 3]
    gpu_ids_string = ', '.join(gpu_ids)
    used_gpus = len(gpu_ids)
    if used_gpus > 1:
        print('Cannot use more than one device.')
        return None
    if cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_string
        dev = device(f'cuda:{gpu_ids_string}')
        print(f'Cuda available, using GPU {gpu_ids_string}')
    else:
        dev = device('cpu')
        print('Cuda NOT available, using CPU!')
    print(f'Created device: {dev}')
    return dev

def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/models/weights/v{version}/{net_type}.pth'

    GPU_IDS = [2]
    device = setup_device(GPU_IDS)
    
    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=device
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict