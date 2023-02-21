import numpy as np


def interpolation(MPR_images, sample_points):
    # MPR_images: [80, 160, 576]
    # sample_points: [576, 2]
    # output: [256, 160, 576]
    img = np.ones([256, 160, 288], dtype=np.float)*-1
    count = 0
    for point_y in range(256):
        for point_x in range(288):
            min_id, min_distance = get_distance_to_curve(np.array([point_x, point_y]), sample_points)
            if abs(min_distance) < 40:
                img[point_y, :, point_x] = MPR_images[min_distance + 40, :, min_id]
                count += 1
    return img


def get_distance_to_curve(point, points):
    diff = points - point
    distance_list = np.sqrt(np.sum((diff)**2, axis=1))
    min_idx = np.argmin(distance_list)
    sign = 1 if points[min_idx, 1] < point[1] else -1
    min_dist = distance_list[min_idx]*sign
    return min_idx, int(min_dist)