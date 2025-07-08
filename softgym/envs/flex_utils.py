import numpy as np
import pyflex
import sys
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from softgym.utils.misc import vectorized_range, vectorized_meshgrid

def rotate_particles(angle):
    r = R.from_euler("zyx", angle, degrees=True)
    pos = pyflex.get_positions().reshape(-1, 4)
    center = np.mean(pos, axis=0)
    pos -= center
    new_pos = pos.copy()[:, :3]
    new_pos = r.apply(new_pos)
    new_pos = np.column_stack([new_pos, pos[:, 3]])
    new_pos += center
    pyflex.set_positions(new_pos)


def move_to_pos(new_pos):
    pos = pyflex.get_positions().reshape(-1, 4)
    center = np.mean(pos, axis=0)
    pos[:, :3] -= center[:3]
    pos[:, :3] += np.asarray(new_pos)
    pyflex.set_positions(pos)


def set_square_scene(config, state=None):
    render_mode = 2
    camera_params = config["camera_params"][config["camera_name"]]
    env_idx = 0
    mass = config["mass"] if "mass" in config else 0.5
    scene_params = np.array(
        [
            *config["ClothPos"],
            *config["ClothSize"],
            *config["ClothStiff"],
            render_mode,
            *camera_params["pos"][:],
            *camera_params["angle"][:],
            camera_params["width"],
            camera_params["height"],
            mass,
            config["flip_mesh"],
        ]
    )

    pyflex.set_scene(env_idx, scene_params, 0)

    if state is not None:
        set_state(state)


def set_cloth3d_scene(config, state=None):
    render_mode = 2
    camera_params = config["camera_params"][config["camera_name"]]
    env_idx = 6
    scene_params = np.concatenate(
        [
            config["pos"][:],
            [config["scale"], config["rot"]],
            config["vel"][:],
            [config["stiff"], config["mass"], config["radius"]],
            camera_params["pos"][:],
            camera_params["angle"][:],
            [camera_params["width"], camera_params["height"]],
            [render_mode],
            [config["cloth_type"]],
            [config["cloth_index"]],
        ]
    )

    pyflex.set_scene(env_idx, scene_params, 0)
    rotate_particles([180, 0, 90])
    move_to_pos([0, 0.05, 0])
    for _ in range(50):
        pyflex.step()

    if state is not None:
        set_state(state)


def set_rope_scene(config, state=None):
    env_idx = 2
    render_mode = 1
    camera_params = config['camera_params'][config['camera_name']]
    scene_params = np.array(
        [*config['init_pos'], config['stretchstiffness'], config['bendingstiffness'], config['radius'], config['segment'], config['mass'], 
            config['scale'], 
            *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], render_mode]
        )

    pyflex.set_scene(env_idx, scene_params, 0)
    num_particles = pyflex.get_n_particles()
    if state is not None:
        set_state(state)




def set_state(state_dict):
    pyflex.set_positions(state_dict["particle_pos"])
    pyflex.set_velocities(state_dict["particle_vel"])
    pyflex.set_shape_states(state_dict["shape_pos"])
    pyflex.set_phases(state_dict["phase"])
    camera_params = deepcopy(state_dict["camera_params"])
    update_camera(camera_params, "default_camera")


def update_camera(camera_params, camera_name="default_camera"):
    camera_param = camera_params[camera_name]
    pyflex.set_camera_params(
        np.array([*camera_param["pos"], *camera_param["angle"], camera_param["width"], camera_param["height"]])
    )


def get_state(camera_params):
    pos = pyflex.get_positions()
    vel = pyflex.get_velocities()
    shape_pos = pyflex.get_shape_states()
    phase = pyflex.get_phases()
    camera_params = deepcopy(camera_params)
    return {
        "particle_pos": pos,
        "particle_vel": vel,
        "shape_pos": shape_pos,
        "phase": phase,
        "camera_params": camera_params,
    }


def get_current_covered_area(cloth_particle_radius= 0.00625, pos=None):
    if pos is None:
        pos = pyflex.get_positions()
    pos = np.reshape(pos, [-1, 4])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(
        np.round((offset[:, 0] - cloth_particle_radius) /
                 span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round(
        (offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(
        np.round((offset[:, 1] - cloth_particle_radius) /
                 span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round(
        (offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    return np.sum(grid) * span[0] * span[1]

