from torch import device, cuda
import os


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

    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_string
    dev = device(f'cuda:{gpu_ids_string}')
    print(f'Using GPU {gpu_ids_string}')
    print(f'Created device: {dev}')
    return dev


def print_memory_summary(dev):
    print(cuda.memory_summary(dev))
