import pandas as pd
import os
import torch
import torchvision

from abc import ABC, abstractmethod  # https://www.python-course.eu/python3_abstract_classes.php
from .envmap import generate_envmap
from numpy import unique
from PIL import Image as PILImage
from tqdm import tqdm

DATA_PATH = '/ivrldata1/students/VIDIT'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
VALIDATION_DATA_PATH = os.path.join(DATA_PATH, 'validate')

ALL_LOCATIONS = ["scene_abandonned_city_54", "scene_artic_mountains_32", "scene_fantasy_castle_59", "scene_grave_43",
                 "scene_medieval_43", "scene_mountains_9", "scene_nordic_23", "scene_road_33", "scene_table_7",
                 "scene_city_24", "scene_fantasy_village_21", "scene_subway_2"]
ALL_DIRECTIONS = ["NW", "N", "NE", "E", "SE", "S", "SW", "W"]
ALL_COLORS = ["2500", "3500", "4500", "5500", "6500"]
ANGLES = {
    'S': 0.,
    'SE': 45.,
    'E': 90.,
    'NE': 135.,
    'N': 180.,
    'NW': 225.,
    'W': 270.,
    'SW': 315.
}


class Sample:
    def __init__(self, path, location, color, direction, scene):
        self.path = path
        self.location = location
        self.color = int(color)
        self.direction = direction
        self.scene = scene
        

class Image:
    def __init__(self, sample, transform):
        self.location = sample.location
        self.color = sample.color
        self.direction = sample.direction
        self.scene = sample.scene
        self.transform = transform
        self.image = self._load_image(sample.path)

    def _load_image(self, file_path):
        img = PILImage.open(file_path)
        return self.transform(img)[:3, :, :]
    
    def as_dict(self):
        dico = {
            'location': self.location,
            'color': self.color,
            'direction': self.dir_to_angle(self.direction),
            'scene': self.scene,
            'image': self.image
        }
        return dico
    
    @staticmethod        
    def dir_to_angle(direction):
        return ANGLES[direction]


class ImageDataset(torch.utils.data.Dataset):
    """
    Base dataset class that needs to be extended with __getitem__ and __len__ functions implementations.
    Provides logic for parsing constructor arguments into lists of samples that can be used as inputs and targets.
    """
    def __init__(self, locations=None, scenes=None, input_directions=None, input_colors=None,
                 target_directions=None, target_colors=None, data_path=TRAIN_DATA_PATH, transform=None):
        self.data_path = data_path

        # Define which images to load
        self.locations = locations if locations else ALL_LOCATIONS
        self.scenes = scenes if scenes else self._load_all_scenes(data_path)
        self.input_directions = input_directions if input_directions else ALL_DIRECTIONS
        self.input_colors = input_colors if input_colors else ALL_COLORS
        self.target_directions = target_directions if target_directions else ALL_DIRECTIONS
        self.target_colors = target_colors if target_colors else ALL_COLORS

        # Load input and target samples
        self.input_samples, self.target_samples = [], []
        for location in tqdm(self.locations):
            if self.exists(data_path, location):
                for color in self.input_colors:
                    for direction in self.input_directions:
                        self.input_samples += self._load_samples(data_path, location, color, direction)
                for color in self.target_colors:
                    for direction in self.target_directions:
                        self.target_samples += self._load_samples(data_path, location, color, direction)

        # Transformations to perform on loaded images
        if transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                transform,
                torchvision.transforms.ToTensor()
            ])
    
    def _load_all_scenes(self, data_path):
        print(" - - Loading all scenes")
        all_scenes = []
        for location in tqdm(self.locations):
            if self.exists(data_path, location):
                file = os.path.join(data_path, location, "dataset.csv")
                all_scenes += pd.read_csv(file)['scene'].drop_duplicates().tolist()
        return all_scenes

    def _load_samples(self, data_path, location, color, direction):
        directory = os.path.join(data_path, location, str(color), direction)
        current_dataset = self.get_current_dataset(data_path, location, color, direction)

        samples = []
        for scene in self.scenes:
            rendered_image_name = self.get_rendered_image_name(current_dataset, scene)
            if rendered_image_name:
                samples.append(Sample(os.path.join(directory, rendered_image_name), location, color, direction, scene))
        return samples
    
    @staticmethod
    def exists(data_path, location):
        return os.path.exists(os.path.join(data_path, location))

    @staticmethod
    def get_current_dataset(data_path, location, color, direction):
        directory = os.path.join(data_path, location, str(color), direction)
        csv_file = os.path.join(directory, f"dataset_{color}_{direction}.csv")
        return pd.read_csv(csv_file)

    @staticmethod
    def get_rendered_image_name(dataset, scene):
        rendered_image = dataset[dataset['scene'] == scene]['rendered_image']
        if len(rendered_image) == 1:
            image_name = rendered_image.values[0]
            return image_name
        elif len(rendered_image) > 1:
            raise RuntimeError(f'Specified scene {scene} exists multiple times in the dataset')

    def __getitem__(self, idx):
        raise NotImplementedError('This class needs to be extended with specific implementation')

    def __len__(self):
        raise NotImplementedError('This class needs to be extended with specific implementation')


class SameTargetSceneDataset(ImageDataset):
    """
    Dataset that loads (input, target) pairs for which both input and target are in the same scene. That means they can
    only differ by light color and direction, but the physical content of both images is exactly the same.
    """
    def __init__(self, locations=None, scenes=None, input_directions=None, input_colors=None,
                 target_directions=None, target_colors=None, data_path=TRAIN_DATA_PATH, transform=None):
        super(SameTargetSceneDataset, self).__init__(locations, scenes, input_directions, input_colors,
                                                     target_directions, target_colors, data_path, transform)
        self.items = []
        for input_sample in self.input_samples:
            for target_sample in self.target_samples:
                # Pair up samples that are from the same scene
                if input_sample['scene'] == target_sample['scene']:
                    self.items.append((input_sample, target_sample))

    def __getitem__(self, idx):
        input_sample, target_sample = self.items[idx]
        return Image(input_sample, self.transform), Image(target_sample, self.transform)

    def __len__(self):
        return len(self.items)


class PairingStrategy(ABC):
    """
    Abstract class to be extend by concrete implementations with sample pairing rules defined in can_be_paired method.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def can_be_paired(self, input_sample, target_sample):
        pass


class DifferentScene(PairingStrategy):
    def __init__(self):
        super(DifferentScene, self).__init__()

    def can_be_paired(self, input_sample, target_sample):
        return input_sample.scene != target_sample.scene
    
    def __repr__(self):
        return 'DifferentScene'


class DifferentLightDirection(PairingStrategy):
    def __init__(self):
        super(DifferentLightDirection, self).__init__()

    def can_be_paired(self, input_sample, target_sample):
        return input_sample.direction != target_sample.direction
    
    def __repr__(self):
        return 'DifferentLightDirection'


class SameScene(PairingStrategy):
    def __init__(self):
        super(SameScene, self).__init__()

    def can_be_paired(self, input_sample, target_sample):
        return input_sample.scene == target_sample.scene
    
    def __repr__(self):
        return 'SameScene'


class SameLightDirection(PairingStrategy):
    def __init__(self):
        super(SameLightDirection, self).__init__()

    def can_be_paired(self, input_sample, target_sample):
        return input_sample.direction == target_sample.direction
    
    def __repr__(self):
        return 'SameLightDirection'


class SameLightColor(PairingStrategy):
    def __init__(self):
        super(SameLightColor, self).__init__()

    def can_be_paired(self, input_sample, target_sample):
        return input_sample.color == target_sample.color
    
    def __repr__(self):
        return 'SameLightColor'


class InputTargetGroundtruthDataset(ImageDataset):
    """
    Dataset that loads ((input, target), ground-truth) tuples.
    In the returned tuple the ground truth image has the same light conditions (direction and color) as target image,
    but was rendered in the same scene as the input image.
    """
    def __init__(self, locations=None, scenes=None, input_directions=None, input_colors=None,
                 target_directions=None, target_colors=None, data_path=TRAIN_DATA_PATH, transform=None,
                 pairing_strategies=None):
        print("Initialising Dataset")
        print("- Loading input and target samples")
        super(InputTargetGroundtruthDataset, self).__init__(locations, scenes, input_directions, input_colors,
                                                            target_directions, target_colors, data_path, transform)
        # Load all ground-truth samples
        print("- Loading ground-truth samples")
        self.pairing_strategies = [] if pairing_strategies is None else pairing_strategies
        self.ground_truth_samples = {}
        for sample in self.target_samples:
            self.ground_truth_samples[(sample.location, sample.color, sample.direction, sample.scene)] = sample
                   
        # Associate (input, target) to correct ground-truth
        print("- Associating input, target and ground-truth samples") 
        self.items = []
        print(len(self.target_samples))

        for input_sample in tqdm(self.input_samples):
            for target_sample in self.target_samples:
                if self._can_be_paired(input_sample, target_sample):
                    ground_truth_sample = self._find_ground_truth_sample_for(input_sample, target_sample)
                    self.items.append(((input_sample, target_sample), ground_truth_sample))

        print("- Finishing initialising Dataset")

    def _can_be_paired(self, input_sample, target_sample):
        for strategy in self.pairing_strategies:
            if not strategy.can_be_paired(input_sample, target_sample):
                return False
        return True

    def _find_ground_truth_sample_for(self, input_sample, target_sample):
        # Localization same as for the input sample
        location = input_sample.location
        scene = input_sample.scene
        # Light conditions of target sample
        color = target_sample.color
        direction = target_sample.direction

        return self.ground_truth_samples[(location, color, direction, scene)]

    def __getitem__(self, idx):
        (x, target), ground_truth = self.items[idx]
        return (Image(x, self.transform).as_dict(), Image(target, self.transform).as_dict()), \
               Image(ground_truth, self.transform).as_dict()

    def __len__(self):
        return len(self.items)


class InputTargetGroundtruthWithGeneratedEnvmapDataset(InputTargetGroundtruthDataset):
    """
    Dataset that loads ((input, input envmap), (target, target envmap), groundtruth) tuples.
    In the returned tuple the ground truth image has the same light conditions (direction and color) as target image,
    but was rendered in the same scene as the input image.
    """
    def __init__(self, locations=None, scenes=None, input_directions=None, input_colors=None,
                 target_directions=None, target_colors=None, data_path=TRAIN_DATA_PATH, transform=None,
                 pairing_strategies=None, envmap_h=16, envmap_w=32):
        super(InputTargetGroundtruthWithGeneratedEnvmapDataset, self).__init__(
            locations, scenes, input_directions, input_colors, target_directions, target_colors, data_path, transform,
            pairing_strategies)

        # Generate ground-truth envmaps
        self.ground_truth_envmaps = self._generate_ground_truth_envmaps(envmap_h, envmap_w)

    @staticmethod
    def _generate_ground_truth_envmaps(envmap_h, envmap_w):
        envmaps = {}
        for direction in ALL_DIRECTIONS:
            for color in ALL_COLORS:
                envmaps[(direction, int(color))] = generate_envmap(direction, int(color), envmap_h, envmap_w)
        return envmaps

    def __getitem__(self, idx):
        (x, target), ground_truth = self.items[idx]
        return (Image(x, self.transform).as_dict(), self.ground_truth_envmaps[(x.direction, x.color)]), \
            (Image(target, self.transform).as_dict(), self.ground_truth_envmaps[(target.direction, target.color)]), \
            Image(ground_truth, self.transform).as_dict()


class OneInputTargetGroundtruthDataset(InputTargetGroundtruthDataset):
    """
    Dataset that loads only one sample of ((image, target), ground-truth) tuple. Can be used to check if network is
    able to overfit given only one sample.
    """
    def __init__(self, locations=None, scenes=None, input_directions=None, input_colors=None,
                 target_directions=None, target_colors=None, data_path=TRAIN_DATA_PATH, transform=None,
                 pairing_strategies=None):
        super(OneInputTargetGroundtruthDataset, self).__init__(locations, scenes, input_directions, input_colors,
                                                               target_directions, target_colors, data_path,
                                                               transform, pairing_strategies)

    def __getitem__(self, idx):
        (x, target), ground_truth = self.items[0]
        return (Image(x, self.transform).as_dict(), Image(target, self.transform).as_dict()), \
               Image(ground_truth, self.transform).as_dict()

    def __len__(self):
        return 1
