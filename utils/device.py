import torch


def setup_device(gpu_id='0'):
    """
    Creates a torch.device
    @param gpu_id: ID of the GPU that should be used if it's available
    @return: device which can be used e.g. for torch.Tensor.to(<device>)
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Cuda available, using GPU {gpu_id}')
    else:
        device = torch.device('cpu')
        print('Cuda NOT available, using CPU!')
    return device
