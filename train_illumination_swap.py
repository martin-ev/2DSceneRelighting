import torch.nn as nn
from torch.optim import Adam

from models.illumination_swap import IlluminationSwapNet
from models.loss import log_l2_loss
from utils.device import setup_device
from utils import storage


# Get used device
GPU_ID = '0'
device = setup_device(GPU_ID)

# Training parameters
NAME = 'Experiment name'
BATCH_SIZE = 5
EPOCHS = 150

# Configure training objects
model = IlluminationSwapNet()
reconstruction_loss = nn.L1Loss()
illumination_loss = log_l2_loss
optimizer = Adam(model.parameters())

# Configure dataloader
dataloader = ...

# Configure tensorboard
...

# Train loop
for epoch in range(1, EPOCHS+1):
    for batch in dataloader:
        ...

# Store trained model
storage.save_trained(model, NAME)
