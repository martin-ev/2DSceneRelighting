import torch
import os


DEFAULT_GPU = 0


def setup_device(gpu_ids=[0]):
    """
    Creates a torch.device
    @param gpu_ids: ID(s) (list of integers) of the GPU(s) that should be used if they're available
    @return: device which can be used e.g. for torch.Tensor.to(<device>)
    """
    gpu_ids = [str(gpu_id) for gpu_id in sorted(gpu_ids) if 0 <= gpu_id <= 3]
    gpu_ids_string = ', '.join(gpu_ids)
    used_gpus = len(gpu_ids)
    if used_gpus > 1:
        print(f"You're about to use {used_gpus} GPUs: {gpu_ids_string}.\n",
              f"To confirm you know what you're doing type 'y':")
        answer = input()
        if answer != 'y':
            print(f'Aborting training on multiple GPUs, choosing GPU {DEFAULT_GPU}')
            gpu_ids_string = f'{DEFAULT_GPU}'
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_string
        device = torch.device('cuda')
        print(f'Cuda available, using GPU: {gpu_ids_string}')
    else:
        device = torch.device('cpu')
        print('Cuda NOT available, using CPU!')
    return device
