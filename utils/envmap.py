from torch import zeros, from_numpy, cat
from math import sqrt, e
from numpy import array


ROTATIONS = {
    'N': 0.0,
    'NE': 0.25,
    'E': 0.5,
    'SE': 0.75,
    'S': 1.0,
    'SW': -0.75,
    'W': -0.5,
    'NW': -0.25
}


# Generated with colour-science, see others/EnvmapGeneration.ipynb
COLOURS = {
    2500: array([255, 222,  76]),
    3500: array([255, 239, 125]),
    4500: array([255, 251, 182]),
    5500: array([238, 255, 234]),
    6500: array([214, 255, 255])
}


def generate_envmap(light_direction, light_temp, height=16, width=32):
    """
    Generates environment map representing the scene with given light properties
    @param light_direction: direction from which the light is coming in the scene
    @param light_temp: (not used; always generates white light) light temperature that corresponds to its color
    @param height: height of the generated environment map
    @param width: width of the generated environment map
    @return: generated environment map with brightness gradient representing the light direction and color set 
    appropriately according to the light temperature
    """
    light_rgb = COLOURS[light_temp]
    centered_envmap = _envmap_with_centered_light(light_rgb, height, width)
    rotated_envmap = _rotate_envmap_to_match_direction(centered_envmap, light_direction)
    return rotated_envmap.view(3 * height * width, 1, 1)


def _envmap_with_centered_light(color, height, width):
    envmap = zeros((3, height, width))
    x_center, y_center = float(width) / 2, float(height) / 2
    sigma = float(min(height, width)) / 10
    sigma2 = 2 * sigma**2
    for x in range(width):
        for y in range(height):
            distance = sqrt((y - y_center) ** 2 + (x - x_center) ** 2)
            fraction = e**(- distance/sigma2)
            envmap[:, y, x] = from_numpy(fraction * color)
    return envmap


def _rotate_envmap_to_match_direction(centered_envmap, light_direction):
    rotation = ROTATIONS[light_direction]
    width = centered_envmap.size()[2]
    max_shift = width // 2
    shift = int(max_shift * rotation)
    return cat((centered_envmap[:, :, -shift:], centered_envmap[:, :, :-shift]), dim=2)



