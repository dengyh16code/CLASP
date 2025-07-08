import sys

sys.path.append("")
import numpy as np
import pyflex
from softgym.action_space.action_space import Picker,PickerPickPlace
from copy import deepcopy
from softgym.envs.flex_utils import set_rope_scene
import cv2
import imageio
from tqdm import tqdm


class RopeEnv:
    def __init__(self, 
                gui=False, 
                dump_visualizations=False, 
                render_dim = 224,
                particle_radius=0.025,
                pick_speed=5e-3, 
                move_speed=5e-3, 
                place_speed=5e-3, 
                lift_height=0.1,
                fling_speed=8e-3,
                ):

        # environment state variables
        self.grasp_states = [False, False]
        self.particle_radius = particle_radius
        self.image_dim = render_dim

        # visualizations
        self.gui = gui
        self.dump_visualizations = dump_visualizations
        self.gui_render_freq = 2
        self.gui_step = 0
        
        self.setup_env()
        
        # primitives parameters
        self.grasp_height = self.action_tool.picker_radius
        self.default_speed = 1e-2
        self.reset_pos = [[0.5, 0.2, 0.5], [-0.5, 0.2, 0.5]]
        self.default_pos = [-0.5, 0.2, 0.5]
        self.pick_speed = pick_speed
        self.move_speed = move_speed
        self.place_speed = place_speed
        self.lift_height = lift_height
        self.fling_speed = fling_speed

        # print("init of rope new env done!")
    
    def setup_env(self):
        pyflex.init(not self.gui, True, 720, 720)
        # self.action_tool = Picker(num_picker=2, picker_radius=0.02, picker_threshold=0.005, 
        #     particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
        self.action_tool = PickerPickPlace(
            num_picker=2,
            particle_radius=0.025,
            picker_threshold=0.005,
            picker_low=(-10.0, 0.0, -10.0),
            picker_high=(10.0, 10.0, 10.0),
        )

        if self.dump_visualizations:
            self.frames = []

    def reset(self, config, state, **kwargs):
        self.current_config = deepcopy(config)
        set_rope_scene(config, state)
        self.camera_params = deepcopy(state["camera_params"])

        self.action_tool.reset(self.reset_pos[0])
        self.step_simulation()
        self.set_grasp(False)
        if self.dump_visualizations:
            self.frames = []

        if bool(kwargs):
            for key, val in kwargs.items():
                if key in ["pick_speed", "move_speed", "place_speed", "lift_height", "fling_speed"]:
                    exec(f"self.{key} = {str(val)}")
        
        self.rope_length = state["rope_length"]
    
    def step_simulation(self):
        pyflex.step()
        if self.gui and self.gui_step % self.gui_render_freq == 0:
            pyflex.render()
        self.gui_step += 1

    def set_grasp(self, grasp):
        self.grasp_states = [grasp] * len(self.grasp_states)

    def render_image(self):
        rgb, depth = pyflex.render()
        rgb = rgb.reshape((720, 720, 4))[::-1, :, :3]
        depth = depth.reshape((720, 720))[::-1]
        rgb = cv2.resize(rgb, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        return rgb, depth

    def render_gif(self, path):
        with imageio.get_writer(path, mode="I", fps=30) as writer:
            for frame in tqdm(self.frames):
                writer.append_data(frame)
    
    def get_rope_keypoints_idx(self):
        num = self.current_config['segment']+1
        indices = [0]
        interval = (num - 2) // 8
        for i in range(1, 9):
            indices.append(i * interval)
        indices.append(num - 1)
        return indices
    
    def get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()


    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    
    
        # pick and drop
    def pick_and_drop(self, pick_pos):
        pick_pos[1] = self.grasp_height
        prepick_pos = pick_pos.copy()
        prepick_pos[1] = self.lift_height

        # execute action
        self.movep([prepick_pos, self.default_pos], speed=0.5)
        self.movep([pick_pos, self.default_pos], speed=0.005) 
        self.set_grasp(True)
        self.movep([prepick_pos, self.default_pos], speed=self.pick_speed)
        self.set_grasp(False)

        # reset
        self.movep(self.reset_pos, speed=0.5)
    
    def grasp_pull(self,pick_pos_left, pick_pos_right, max_dis = 0.9, increment_step = 0.02):
        pick_pos_left[1] = self.grasp_height
        pick_pos_right[1] = self.grasp_height

        prepick_pos_left = pick_pos_left.copy()
        prepick_pos_left[1] = self.lift_height

        prepick_pos_right = pick_pos_right.copy()
        prepick_pos_right[1] = self.lift_height

        # grasp rope
        self.movep([prepick_pos_left, prepick_pos_right])
        self.movep([pick_pos_left, pick_pos_right])
        self.set_grasp(True)

        #pull rope
        midpoint = (pick_pos_left + pick_pos_right)/2
        direction = pick_pos_left - pick_pos_right
        direction = direction / np.linalg.norm(direction)
        self.movep([pick_pos_left, pick_pos_right], speed=5e-4, min_steps=20)
        
        curr_length = self.get_endpoint_distance()
        pull_dis = 0
        while curr_length <= 1.2*self.rope_length:
            pull_dis += increment_step
            left = pick_pos_left + direction*pull_dis/2
            right = pick_pos_right - direction*pull_dis/2
            self.movep([left, right], speed=5e-4)
            if pull_dis > max_dis:
                print("fuck")
                return pull_dis
                
            curr_length = self.get_endpoint_distance()
            # print(curr_length)
            # print(self.rope_length)

        
        
        #release
        # self.set_grasp(False)

        # # reset
        # self.movep(self.reset_pos, speed=0.5)

    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.1
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0]
            deltas = [(targ - curr) for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and\
                    (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr+delta*speed), float(gs)])
    
            action = np.array(action)
            self.action_tool.step(action, step_sim_fn=self.step_simulation)
            if self.dump_visualizations:
                self.frames.append(self.render_image()[0])

    
    def get_keypoints(self,keypoints_index):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[keypoints_index, :3]
        return keypoint_pos


