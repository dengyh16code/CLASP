import numpy as np
import cv2
from softgym.utils.gemo_utils import *

def get_matrix_world_to_camera(camera_params):
    cam_x, cam_y, cam_z = (
        camera_params["default_camera"]["pos"][0],
        camera_params["default_camera"]["pos"][1],
        camera_params["default_camera"]["pos"][2],
    )
    cam_x_angle, cam_y_angle, cam_z_angle = (
        camera_params["default_camera"]["angle"][0],
        camera_params["default_camera"]["angle"][1],
        camera_params["default_camera"]["angle"][2],
    )

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(-cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(-cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = -cam_x
    translation_matrix[1][3] = -cam_y
    translation_matrix[2][3] = -cam_z

    return rotation_matrix @ translation_matrix


def get_pixel_coord_from_world(coord, rgb_shape, camera_params):
    matrix_world_to_camera = get_matrix_world_to_camera(camera_params)
    height, width = rgb_shape

    coord = np.array([coord])
    world_coordinate = np.concatenate([coord, np.ones((len(coord), 1))], axis=1)
    camera_coordinate = matrix_world_to_camera @ world_coordinate.T
    camera_coordinate = camera_coordinate.T
    K = intrinsic_from_fov(height, width, 45)

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = x * fx / depth + u0
    v = y * fy / depth + v0

    pixel = np.array([u, v]).squeeze(1)

    return pixel


def get_world_coord_from_pixel(pixel, depth, camera_params):
    matrix_world_to_camera = get_matrix_world_to_camera(camera_params)
    matrix_camera_to_world = np.linalg.inv(matrix_world_to_camera)
    height, width = depth.shape
    K = intrinsic_from_fov(height, width, 45)

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    u, v = pixel[0], pixel[1]
    z = depth[int(np.rint(u)), int(np.rint(v))]
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy

    cam_coord = np.ones(4)
    cam_coord[:3] = (x, y, z)
    world_coord = matrix_camera_to_world @ cam_coord

    return world_coord[:3]


def get_world_coords_from_pixels(depth, camera_params):
    height, width = depth.shape
    K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

    # Apply back-projection: K_inv @ pixels * depth
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x = np.linspace(0, width - 1, width).astype(np.float)
    y = np.linspace(0, height - 1, height).astype(np.float)
    u, v = np.meshgrid(x, y)
    one = np.ones((height, width, 1))
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.dstack([x, y, z, one])

    matrix_world_to_camera = get_matrix_world_to_camera(camera_params)

    # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
    world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
    world_coords = world_coords.transpose().reshape((height, width, 4))

    return world_coords


#################################################
################# visualization#####################
#################################################
# def action_viz(img, pick, place):
#     cv2.circle(img, (int(pick[0]), int(pick[1])), 3, (0, 0, 0), 2)
#     cv2.arrowedLine(img, (int(pick[0]), int(pick[1])), (int(place[0]), int(place[1])), (0, 0, 0), 2)
#     return img


def action_viz(img, pick, place, primitive):
    if primitive == "single":
        cv2.circle(img, (int(pick[0]), int(pick[1])), 3, (0, 0, 0), 2)
        cv2.arrowedLine(img, (int(pick[0]), int(pick[1])), (int(place[0]), int(place[1])), (0, 0, 0), 2)
    elif primitive == "dual" or primitive == "place" or primitive == "hang":
        pick_left, pick_right = pick
        place_left, place_right = place
        cv2.circle(img, (int(pick_left[0]), int(pick_left[1])), 3, (0, 0, 0), 2)
        cv2.arrowedLine(img, (int(pick_left[0]), int(pick_left[1])), (int(place_left[0]), int(place_left[1])), (1, 0, 0), 2)
        cv2.circle(img, (int(pick_right[0]), int(pick_right[1])), 3, (0, 0, 0), 2)
        cv2.arrowedLine(img, (int(pick_right[0]), int(pick_right[1])), (int(place_right[0]), int(place_right[1])), (0, 1, 0), 2)
    elif primitive == "flatten":
        pick_left, pick_right = pick
        cv2.circle(img, (int(pick_left[0]), int(pick_left[1])), 3, (0, 0, 0), 2)
        cv2.circle(img, (int(pick_right[0]), int(pick_right[1])), 3, (0, 0, 0), 2)

    return img


#################################################
################# mask###########################
#################################################
def nearest_to_mask(u, v, depth):
    mask_idx = np.argwhere(depth)
    nearest_idx = mask_idx[((mask_idx - [u, v]) ** 2).sum(1).argmin()]
    return nearest_idx
