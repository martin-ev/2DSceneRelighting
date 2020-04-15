import torch


TRAINED_MODELS_PATH = "/ivrldata1/students/team6/2DSceneRelighting/trained_models"


def build_full_path(name):
    return TRAINED_MODELS_PATH+"/"+name+".pth"


def save_trained(model, name):
    torch.save(model.state_dict(), build_full_path(name))


def load_trained(model, name):
    model.load_state_dict(torch.load(build_full_path(name)))
