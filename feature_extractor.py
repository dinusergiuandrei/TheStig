import numpy as np

red = [255, 0, 0]
grey = [150, 150, 150]
green = [70, 230, 70]
colors = np.array([red, grey, green])

"""
The car is usually in the box:
(68, 46)    (68, 49)
(75, 46)    (75, 49)

Road width 17 pixels 40 - 56
The columns taken into consideration are:

46-17 -> 29
46+17 -> 63

The features will be distances from the height 67, going up until the first green pixel
"""


def get_color_distance(p1, p2):
    # return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])
    return np.linalg.norm(p1 - p2)


def get_replacement_index(p):
    diffs = np.zeros(len(colors))
    for i in range(len(colors)):
        diffs[i] = get_color_distance(p, colors[i])
    return np.argmin(diffs)


def get_replacement(p):
    return colors[get_replacement_index(p)]


def get_replacement_image(image):
    (h, w, p) = image.shape
    for i in range(h):
        for j in range(w):
            image[i][j] = get_replacement(image[i][j])
    return image


def simplify_to_2d(image):
    (h, w, p) = image.shape
    result = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            result[i][j] = get_replacement_index(image[i][j])
    return result


def get_feature(image, column):
    start = 67
    r = get_replacement_index(image[start][column])
    while start >= 1 and r != 2:
        start -= 1
        r = get_replacement_index(image[start][column])
    height = 67 - start
    return height / 68.0


def get_reward(image, action):
    left_distance = 0
    right_distance = 0
    line = 65
    left_start = 46
    right_start = 49
    r = get_replacement_index(image[line][left_start])
    while left_start > 0 and r != 2:
        left_start -= 1
        left_distance += 1
        r = get_replacement_index(image[line][left_start])

    r = get_replacement_index(image[line][right_start])
    while right_start < 95 and r != 2:
        right_start += 1
        right_distance += 1
        r = get_replacement_index(image[line][right_start])

    far_on_grass = False
    if left_distance == 0 and right_distance == 0:
        grass_left = 0
        grass_right = 0
        left_start = 46
        right_start = 49
        r = get_replacement_index(image[line][left_start])
        while left_start > 0 and r == 2:
            left_start -= 1
            grass_left += 1
            r = get_replacement_index(image[line][left_start])

        r = get_replacement_index(image[line][right_start])
        while right_start < 95 and r == 2:
            right_start += 1
            grass_right += 1
            r = get_replacement_index(image[line][right_start])
        if min(grass_left, grass_right) > 15:
            far_on_grass = True

    if left_distance == 0 and right_distance == 0:  # all wheels on grass
        return 0, far_on_grass
    if left_distance == 0 and action[0] == -1.0:
        return 0, far_on_grass
    if right_distance == 0 and action[0] == 1.0:
        return 0, far_on_grass
    if left_distance == 0 or right_distance == 0:  # only one side of wheels on grass
        return 0.1, far_on_grass
    return (20 - abs(right_distance - left_distance)) / 21, far_on_grass


def get_features(image):
    # magic numbers are explained higher
    left = 29
    right = 64
    features = np.zeros(right - left)
    for column in range(left, right):
        features[column - left] = get_feature(image, column)
    return features
