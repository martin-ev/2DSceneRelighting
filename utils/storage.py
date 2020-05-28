import torch
import os


TRAINED_MODELS_PATH = "/ivrldata1/students/team6/2DSceneRelighting/trained_models"
CHECKPOINT_PATH = "/ivrldata1/students/team6/2DSceneRelighting/checkpoints"


def build_trained_model_path(name):
    return os.path.join(TRAINED_MODELS_PATH, name + '.pth')


def build_checkpoint_path(name):
    return os.path.join(CHECKPOINT_PATH, name + '.pth')


def save_trained(model, name):
    torch.save(model.state_dict(), build_trained_model_path(name))


def save_checkpoint(model_state_dict, optimizer_state_dict, name):
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict
    }, build_checkpoint_path(name))


def load_checkpoint(name):
    return torch.load(build_checkpoint_path(name))


def load_trained(model, name):
    model.load_state_dict(torch.load(build_trained_model_path(name)))
    return model
