from torch import zeros, tensor, cat
from math import sqrt, e
from numpy import array
from kornia.color import hsv_to_rgb


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


# Generated with colour-science and colorsys, see others/EnvmapGeneration.ipynb
COLORS_HSV = {
    2500: array([14, 231, 255]),
    3500: array([19, 192, 255]),
    4500: array([22, 132, 255]),
    5500: array([24, 65, 255]),
    6500: array([54, 0, 255])
}


def generate_envmap(light_direction, light_temp, mode='hsv', height=16, width=32):
    """
    Generates environment map representing the scene with given light properties
    @param light_direction: direction from which the light is coming in the scene
    @param light_temp: light temperature that corresponds to its color
    @param mode: colorspace in which the envmap should be generated (RGB or HSV).
    @param height: height of the generated environment map
    @param width: width of the generated environment map
    @return: generated environment map with brightness gradient representing the light direction and color set 
    appropriately according to the light temperature
    """
    light_hsv = COLORS_HSV[light_temp]
    centered_envmap = _envmap_with_centered_light(light_hsv, height, width)
    rotated_envmap = _rotate_envmap_to_match_direction(centered_envmap, light_direction)
    if mode == 'rgb':
        return (255. * hsv_to_rgb(rotated_envmap / 255.)).view(3 * height * width, 1, 1)
    elif mode == 'hsv':
        hue, saturation = rotated_envmap[0, 0, 0], rotated_envmap[1, 0, 0]
        # unsqueezing to multiple dimensions: https://github.com/pytorch/pytorch/issues/9410#issuecomment-404968513
        return cat((hue[(None,)*3],
                    saturation[(None,)*3],
                    rotated_envmap[2, :, :].view(height * width, 1, 1)), dim=0)


def _envmap_with_centered_light(color, height, width):
    envmap = zeros((3, height, width))
    x_center, y_center = float(width) / 2, float(height) / 2
    sigma = float(min(height, width)) / 10
    sigma2 = 2 * sigma**2
    for x in range(width):
        for y in range(height):
            distance = sqrt((y - y_center) ** 2 + (x - x_center) ** 2)
            fraction = e**(- distance/sigma2)
            envmap[:, y, x] = tensor([color[0], color[1], int(fraction * color[2])])

    return envmap


def _rotate_envmap_to_match_direction(centered_envmap, light_direction):
    rotation = ROTATIONS[light_direction]
    width = centered_envmap.size()[2]
    max_shift = width // 2
    shift = int(max_shift * rotation)
    return cat((centered_envmap[:, :, -shift:], centered_envmap[:, :, :-shift]), dim=2)
