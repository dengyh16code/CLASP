import sys

sys.path.append("")
import numpy as np
import pyflex
from softgym.action_space.action_space import PickerPickPlace
from softgym.envs.flex_utils import set_cloth3d_scene, set_square_scene
from softgym.utils.misc import rotate_rigid_object, quatFromAxisAngle
from copy import deepcopy
import cv2
import imageio
from tqdm import tqdm
import copy

class ClothEnv:
    def __init__(self, 
                gui=False, 
                dump_visualizations=False, 
                cloth3d=True, 
                pick_speed=5e-3, 
                move_speed=5e-3, 
                place_speed=5e-3, 
                lift_height=0.1,
                fling_speed=8e-3,
                render_dim=224, 
                particle_radius=0.00625,
                # forward_ratio=0.5
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

        # setup env
        self.cloth3d = cloth3d
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
        # self.forward_ratio = forward_ratio
        # self.lift_ratio = 1 if forward_ratio < 0.5 else 2

    def setup_env(self):
        pyflex.init(not self.gui, True, 720, 720)
        self.action_tool = PickerPickPlace(
            num_picker=2,
            particle_radius=self.particle_radius,
            picker_threshold=0.005,
            picker_low=(-10.0, 0.0, -10.0),
            picker_high=(10.0, 10.0, 10.0),
        )
        if self.dump_visualizations:
            self.frames = []

    def reset(self, config, state, **kwargs):
        self.current_config = deepcopy(config)
        if self.cloth3d:
            set_cloth3d_scene(config=config, state=state)
        else:
            set_square_scene(config=config, state=state)
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
        
        self.max_area = state["max_area"]
    

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

    #################################################
    ######################Picker#####################
    #################################################
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

    # single arm primitive, default use picker1 for manipulation
    def pick_and_place_single(self, pick_pos, place_pos):
        pick_pos[1] = self.grasp_height
        place_pos[1] = self.grasp_height

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = self.lift_height

        preplace_pos = place_pos.copy()
        preplace_pos[1] = self.lift_height

        # execute action
        self.movep([prepick_pos, self.default_pos], speed=0.5)
        self.movep([pick_pos, self.default_pos], speed=0.005) 
        self.set_grasp(True)
        self.movep([prepick_pos, self.default_pos], speed=self.pick_speed)
        self.movep([preplace_pos, self.default_pos], speed=self.move_speed)
        self.movep([place_pos, self.default_pos], speed=self.place_speed)
        self.set_grasp(False)
        self.movep([preplace_pos, self.default_pos], speed=0.5)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    # pick and drop
    def pick_and_drop(self, pick_pos):
        pick_pos[1] = self.grasp_height
        prepick_pos = pick_pos.copy()
        prepick_pos[1] = 1.25 #self.lift_height

        # execute action
        self.movep([prepick_pos, self.default_pos], speed=0.5)
        self.movep([pick_pos, self.default_pos], speed=0.005) 
        self.set_grasp(True)
        self.movep([prepick_pos, self.default_pos], speed=self.pick_speed)
        self.set_grasp(False)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    # dual arm primitive
    def pick_and_place_dual(self, pick_pos_left, place_pos_left, pick_pos_right, place_pos_right):
        pick_pos_left[1] = self.grasp_height
        place_pos_left[1] = self.grasp_height        
        pick_pos_right[1] = self.grasp_height
        place_pos_right[1] = self.grasp_height

        prepick_pos_left = pick_pos_left.copy()
        prepick_pos_left[1] = self.lift_height
        prepick_pos_right = pick_pos_right.copy()
        prepick_pos_right[1] = self.lift_height

        preplace_pos_left = place_pos_left.copy()
        preplace_pos_left[1] = self.lift_height
        preplace_pos_right = place_pos_right.copy()
        preplace_pos_right[1] = self.lift_height

        # execute action
        self.movep([prepick_pos_left, prepick_pos_right], speed=0.5)
        self.movep([pick_pos_left, pick_pos_right], speed=0.005) 
        self.set_grasp(True)
        self.movep([prepick_pos_left, prepick_pos_right], speed=self.pick_speed)
        self.movep([preplace_pos_left, preplace_pos_right], speed=self.move_speed)
        self.movep([place_pos_left, place_pos_right], speed=self.place_speed)
        self.set_grasp(False)
        self.movep([preplace_pos_left, preplace_pos_right], speed=0.5)

        # reset
        self.movep(self.reset_pos, speed=0.5)

    def pick_and_fling(self, pick_pos_left, pick_pos_right):
        pick_pos_left[1] = self.grasp_height
        pick_pos_right[1] = self.grasp_height

        prepick_pos_left = pick_pos_left.copy()
        prepick_pos_left[1] = self.lift_height

        prepick_pos_right = pick_pos_right.copy()
        prepick_pos_right[1] = self.lift_height

        # grasp distance
        dist = np.linalg.norm(np.array(prepick_pos_left) - np.array(prepick_pos_right))
        
        # pick cloth
        self.movep([prepick_pos_left, prepick_pos_right])
        self.movep([pick_pos_left, pick_pos_right])
        self.set_grasp(True)

        # prelift & stretch
        self.movep([[-dist / 2, 0.3, 0], [dist / 2, 0.3, 0]], speed= 0.1) #5e-3
        if not self.is_cloth_grasped():
            return False
        dist = self.stretch_cloth(grasp_dist=dist, max_grasp_dist=0.4, fling_height=0.5)

        # lift
        fling_height = self.lift_cloth(grasp_dist=dist, fling_height=0.5)
        
        # fling
        self.fling(dist=dist, fling_height=fling_height, fling_speed=self.fling_speed)

        # reset
        self.movep(self.reset_pos, speed=0.5)


    def fling(self, dist, fling_height, fling_speed):
        # fling
        self.movep([[-dist/2, fling_height, -0.2],
                    [dist/2, fling_height, -0.2]], speed=fling_speed)
        self.movep([[-dist/2, fling_height, 0.2],
                    [dist/2, fling_height, 0.2]], speed=fling_speed)
        self.movep([[-dist/2, fling_height, 0.2],
                    [dist/2, fling_height, 0.2]], speed=1e-2, min_steps=4)
        
        # lower & flatten
        self.movep([[-dist/2, self.grasp_height*2, 0.2],
                    [dist/2, self.grasp_height*2, 0.2]], speed=fling_speed)

        self.movep([[-dist/2, self.grasp_height, 0],
                    [dist/2, self.grasp_height, 0]], speed=fling_speed)
        
        self.movep([[-dist/2, self.grasp_height, -0.2],
                    [dist/2, self.grasp_height, -0.2]], speed= 1e-2)  #5e-3
        
        # release
        self.set_grasp(False)

        # if self.dump_visualizations:
        self.movep(
                [[-dist/2, fling_height, -0.2],
                 [dist/2, fling_height, -0.2]], min_steps=10)
    
    
    def place_on_rigid_obj(self, pick_pos_left, pick_pos_right, place_pos, place_angle, place_height, forward_ratio):    
        pick_pos_left[1] = self.grasp_height
        pick_pos_right[1] = self.grasp_height

        prepick_pos_left = [pick_pos_left[0], self.lift_height, pick_pos_left[2]]
        prepick_pos_right = [pick_pos_right[0], self.lift_height, pick_pos_right[2]]

        # grasp distance
        dist = np.linalg.norm(np.array(prepick_pos_left) - np.array(prepick_pos_right))
      
        # pick cloth
        self.movep([prepick_pos_left, prepick_pos_right])
        self.movep([pick_pos_left, pick_pos_right])
        self.set_grasp(True)

        # lift and higher than rigid_object
        lift_height = 0.4
        lift_height = self.lift_cloth_p(pick_pos_left,pick_pos_right,lift_height)
        lift_pos_left = [pick_pos_left[0], lift_height + place_height, pick_pos_left[2]]
        lift_pos_right = [pick_pos_right[0], lift_height + place_height, pick_pos_right[2]]
        self.movep([lift_pos_left,lift_pos_right], speed=0.01)
        
        # place on rigid_object
        forward_dis = dist*forward_ratio
        lift_ratio = 1 if forward_ratio < 0.5 else 2
        
        #center point
        place_pos_left = [place_pos[0]-dist*np.cos(place_angle)/2, place_height+lift_height/lift_ratio,  place_pos[2]+dist*np.sin(place_angle)/2]
        place_pos_right = [place_pos[0]+dist*np.cos(place_angle)/2, place_height+lift_height/lift_ratio, place_pos[2]-dist*np.sin(place_angle)/2]

        #pre place point
        place_pos_left_pre = [place_pos_left[0]+forward_dis*np.sin(place_angle), place_height+lift_height, place_pos_left[2]+forward_dis*np.cos(place_angle)]
        place_pos_right_pre = [place_pos_right[0]+forward_dis*np.sin(place_angle), place_height+lift_height, place_pos_right[2]+forward_dis*np.cos(place_angle)]

        #final place point
        place_pos_left_final = [place_pos_left[0]-forward_dis*np.sin(place_angle), place_height, place_pos_left[2]-forward_dis*np.cos(place_angle)]
        place_pos_right_final = [place_pos_right[0]-forward_dis*np.sin(place_angle), place_height, place_pos_right[2]-forward_dis*np.cos(place_angle)]

        #prepare
        self.movep([place_pos_left_pre, place_pos_right_pre], speed=5e-3)
        
        #lower & flatten
        self.movep([place_pos_left, place_pos_right], speed=5e-3)
        self.movep([place_pos_left_final, place_pos_right_final], speed=5e-3)

        #release
        self.set_grasp(False)

        if self.dump_visualizations:
            self.movep(
                [[-dist/2, self.grasp_height*2, -0.2],
                 [dist/2, self.grasp_height*2, -0.2]], min_steps=10)

        # reset
        self.movep(self.reset_pos, speed=0.5)

        return place_pos_left, place_pos_right

    
    def stretch_cloth(self, grasp_dist, fling_height=0.7, max_grasp_dist=0.7, increment_step=0.02):
        # lift cloth in the air
        left, right = self.action_tool._get_pos()[0]
        left[1] = fling_height
        right[1] = fling_height
        midpoint = (left + right)/2
        direction = left - right
        if np.linalg.norm(direction) < 1e-4:
            return 0.01
        direction = direction / np.linalg.norm(direction)
        self.movep([left, right], speed=5e-4, min_steps=20)
        stable_steps = 0
        cloth_midpoint = 1e2
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            # get midpoints
            high_positions = positions[positions[:, 1] > fling_height-0.1, ...]
            if (high_positions[:, 0] < 0).all() or \
                    (high_positions[:, 0] > 0).all():
                # single grasp
                return grasp_dist
            positions = [p for p in positions]
            positions.sort(
                key=lambda pos: np.linalg.norm(pos[[0, 2]]-midpoint[[0, 2]]))
            new_cloth_midpoint = positions[0]
            stable = np.linalg.norm(
                new_cloth_midpoint - cloth_midpoint) < 1.5e-2
            if stable:
                stable_steps += 1
            else:
                stable_steps = 0
            stretched = stable_steps > 2
            if stretched:
                return grasp_dist
            cloth_midpoint = new_cloth_midpoint
            grasp_dist += increment_step
            left = midpoint + direction*grasp_dist/2
            right = midpoint - direction*grasp_dist/2
            self.movep([left, right], speed=5e-4)
            if grasp_dist > max_grasp_dist:
                return max_grasp_dist


    def lift_cloth(self, grasp_dist, fling_height: float = 0.7, increment_step: float = 0.05, max_height=0.7):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            heights = positions[:, 1]
            if heights.min() > 0.02:
                return fling_height
            fling_height += increment_step
            self.movep([[-grasp_dist/2, fling_height, -0.3],
                        [grasp_dist/2, fling_height, -0.3]], speed=1e-3)
            if fling_height >= max_height:
                return fling_height
    
    def lift_cloth_p(self, left_position, right_position, fling_height: float = 0.7, increment_step: float = 0.05, max_height=0.7):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:, :3]
            heights = positions[:, 1]
            if heights.min() > 0.02:
                return fling_height
            fling_height += increment_step
            self.movep([[left_position[0], fling_height,left_position[2]],
                        [right_position[0], fling_height, right_position[2]]], speed=0.01)
            if fling_height >= max_height:
                return fling_height


    #################################################
    ###################Ground Truth##################
    #################################################
    # square cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    # Cloth Keypoints are defined:
    #  0  1  2
    #  3  4  5
    #  6  7  8
    def get_square_keypoints_idx(self):
        """The keypoints are defined as the four corner points of the cloth"""
        dimx, dimy = self.current_config["ClothSize"]
        idx0 = 0
        idx1 = int((dimx - 1) / 2)
        idx2 = dimx - 1
        idx3 = int((dimy - 1) / 2) * dimx
        idx4 = int((dimy - 1) / 2) * dimx + int((dimx - 1) / 2)
        idx5 = int((dimy - 1) / 2) * dimx + dimx - 1
        idx6 = dimx * (dimy - 1)
        idx7 = dimx * (dimy - 1) + int((dimx - 1) / 2)
        idx8 = dimx * dimy - 1
        return [idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8]

    def get_keypoints(self,keypoints_index):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[keypoints_index, :3]
        return keypoint_pos

    def is_cloth_grasped(self):
        positions = pyflex.get_positions().reshape((-1, 4))
        positions = positions[:, :3]
        heights = positions[:, 1]
        return heights.max() > 0.2


    #################################################
    ###################rigid object##################
    #################################################
    def change_rigid_obj_state(self, rigid_state):    
        tool_state = copy.deepcopy(np.array(pyflex.get_shape_states()).reshape(-1, 14)[:2, :])    
        state_tool_box = np.concatenate((tool_state,rigid_state),axis=0)
        pyflex.set_shape_states(state_tool_box)
        self.step_simulation()
    
    def create_glass(self, glass_dis_x, glass_dis_z, height, border):
        """
        the glass is a box, with each wall of it being a very thin box in Flex.
        dis_x: the length of the glass
        dis_z: the width of the glass
        height: the height of the glass.
        border: the thickness of the glass wall.
        """
        self.dim_shape_state = 14
        self.border = border
        self.height = height
        self.glass_dis_x = glass_dis_x
        self.glass_dis_z = glass_dis_z

        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        boxes = []

        # floor
        halfEdge = np.array([glass_dis_x / 2. + border, border / 2., glass_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # left wall
        halfEdge = np.array([border / 2., (height) / 2., glass_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # right wall
        boxes.append([halfEdge, center, quat])

        # back wall
        halfEdge = np.array([(glass_dis_x) / 2., (height) / 2., border / 2.])
        boxes.append([halfEdge, center, quat])

        # front wall
        boxes.append([halfEdge, center, quat])

        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)

        return boxes
    

    def create_hanger(self, glass_dis_x, glass_dis_z, height, border):
        """
        create a hanger
        """
        self.dim_shape_state = 14
        self.border = border
        self.height = height
        self.glass_dis_x = glass_dis_x
        self.glass_dis_z = glass_dis_z

        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        boxes = []

        # top
        halfEdge = np.array([glass_dis_x / 2. + border, border / 2., border / 2.])
        boxes.append([halfEdge, center, quat])

        # left bottom
        halfEdge = np.array([border / 2., border / 2., glass_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # right bottom
        boxes.append([halfEdge, center, quat])

        # left height
        halfEdge = np.array([border / 2., (height) / 2., border / 2.])
        boxes.append([halfEdge, center, quat])

        # right height
        boxes.append([halfEdge, center, quat])

        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)

        return boxes
    
    
    def init_glass_state(self, x_center, z_center, theta):
        '''
        set the initial state of the glass in 2D space..
        '''
        dis_x, dis_z = self.glass_dis_x, self.glass_dis_z
        quat = quatFromAxisAngle([0, 1, 0.], theta)

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        # floor 
        rotate_center = np.array([x_center, 0, z_center])
        states[0, :3] = rotate_center
        states[0, 3:6] = rotate_center

        # left wall
        relative_coord = np.array([-(dis_x+ self.border) / 2., (self.height) / 2., 0.])
        left_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[1, :3] = left_wall_center
        states[1, 3:6] = left_wall_center

        # right wall
        relative_coord = np.array([(dis_x+ self.border) / 2., (self.height) / 2., 0.])
        right_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[2, :3] = right_wall_center
        states[2, 3:6] = right_wall_center

        # back wall
        relative_coord = np.array([0, (self.height) / 2., -(dis_z+ self.border) / 2.])
        back_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[3, :3] = back_wall_center
        states[3, 3:6] = back_wall_center

        # front wall
        relative_coord = np.array([0, (self.height) / 2., (dis_z+ self.border) / 2.])
        front_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[4, :3] = front_wall_center
        states[4, 3:6] = front_wall_center

        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states
    
    
    def init_hanger_state(self, x_center, z_center, theta):
        '''
        set the initial state of the hanger in 2D space..
        '''
        dis_x = self.glass_dis_x
        quat = quatFromAxisAngle([0, 1, 0.], theta)

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        # top 
        rotate_center = np.array([x_center, self.height+ 1.5*self.border, z_center])
        states[0, :3] = rotate_center
        states[0, 3:6] = rotate_center

        # left bottom
        relative_coord = np.array([-(dis_x+ self.border) / 2., -(self.height+ self.border), 0.])
        left_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[1, :3] = left_wall_center
        states[1, 3:6] = left_wall_center

        # right bottom
        relative_coord = np.array([(dis_x+ self.border) / 2., -(self.height+ self.border), 0.])
        right_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[2, :3] = right_wall_center
        states[2, 3:6] = right_wall_center

        # left height
        relative_coord = np.array([-(dis_x+ self.border) / 2., -(self.height+ self.border)/2.0, 0.])
        back_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[3, :3] = back_wall_center
        states[3, 3:6] = back_wall_center

        # right height
        relative_coord = np.array([(dis_x+ self.border) / 2., -(self.height+ self.border)/2.0, 0.])
        front_wall_center = rotate_rigid_object(center=rotate_center, axis=np.array([0, 1, 0]), angle=theta, relative=relative_coord)
        states[4, :3] = front_wall_center
        states[4, 3:6] = front_wall_center

        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states
    


    #################################################
    #################action primitives###############
    #################################################
    # def grasp(self, grasp_pos_left, grasp_pos_right, dual=False):
        
    #     grasp_pos_left[1] = self.grasp_height
    #     grasp_pos_right[1] = self.grasp_height

    #     pregrasp_pos_left = grasp_pos_left.copy()
    #     pregrasp_pos_left[1] = self.lift_height
        
    #     pregrasp_pos_right = grasp_pos_right.copy()
    #     pregrasp_pos_right[1] = self.lift_height      
   
    #     # execute action
    #     if dual:
    #         self.movep([pregrasp_pos_left, pregrasp_pos_right], speed=0.5)
    #         self.movep([grasp_pos_left, grasp_pos_right], speed=0.005) 
    #         self.set_grasp(True)
    #         self.movep([pregrasp_pos_left, pregrasp_pos_right], speed=self.pick_speed)
    #     else:
    #         self.movep([pregrasp_pos_left, self.default_pos], speed=0.5)
    #         self.movep([grasp_pos_left, self.default_pos], speed=0.005) 
    #         self.set_grasp(True)
    #         self.movep([pregrasp_pos_left, self.default_pos], speed=self.pick_speed)
    
         
    # def move_to(self, pos_left, pos_right, dual=False):
    #     pos_left[1] = self.grasp_height
    #     pos_right[1] = self.grasp_height
        
    #     pre_pos_left = pos_left.copy()
    #     pre_pos_left[1] = self.lift_height
        
    #     pre_pos_right = pos_right.copy()
    #     pre_pos_right[1] = self.lift_height
    
    #     if dual:
    #         self.movep([pre_pos_left, pre_pos_right], speed=self.move_speed)
    #         self.movep([pos_left, pos_right], speed=self.place_speed)
    #     else:
    #         self.movep([pre_pos_left, self.default_pos], speed=self.move_speed)
    #         self.movep([pos_left, self.default_pos], speed=self.place_speed)


    # def press(self,pos_left, pos_right, dual=False):
    #     pre_pos_left = pos_left.copy()
    #     pre_pos_left[1] = self.lift_height
    #     pre_pos_right = pos_right.copy()
    #     pre_pos_right[1] = self.lift_height
    #     self.movep([pre_pos_left, pre_pos_right], speed=0.5)
    #     # reset
    #     self.movep(self.reset_pos, speed=0.5)
    
    # def release(self):
    #     self.set_grasp(False)

    
   


    
    
    

    

