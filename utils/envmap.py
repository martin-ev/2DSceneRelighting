from torch import zeros, from_numpy, cat
from math import sqrt, e
from colour import CCT_to_xy, xy_to_XYZ, XYZ_to_RGB
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

# conversion matrix from color-science example:
# https://www.colour-science.org/api/0.3.6/html/colour.models.rgb.html?highlight=primary
XYZ_to_RGB_matrix = array([
    [3.24100326, -1.53739899, -0.49861587],
    [-0.96922426, 1.87592999, 0.04155422],
    [0.05563942, -0.20401120, 1.05714897]
])

# Inverse of RGB to XYZ matrix from lecture notes
# array([
#     [2.36461, -0.896541, -0.468073],
#     [-0.515166, 1.42641, 0.0887581],
#     [0.0052037, -0.0144082, 1.0092]
# ])


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
    light_rgb = _light_temperature_to_rgb(light_temp)
    print(light_rgb)
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


def _light_temperature_to_rgb(cct):
    xy = CCT_to_xy(cct)
    print(xy)
    xyz = xy_to_XYZ(xy)
    rgb = XYZ_to_RGB(xyz, illuminant_XYZ=xy, illuminant_RGB=xy, XYZ_to_RGB_matrix=XYZ_to_RGB_matrix)
    return (255 * rgb).clip(0, 255)



