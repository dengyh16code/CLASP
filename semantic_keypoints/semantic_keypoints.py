import sys
sys.path.append("/home/adacomp/CLASP")
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.process import resize, mask_pose_alignment, center_object, dino_simularity_with_bbs
from utils.gpt_utils import request_gpt,load_prompts
from utils.visual_prompt_utils import annotate_grid, annotate_grid_with_kp
import copy
import os
import json
from cloud_services.apis.owlv2 import OWLViT
from cloud_services.apis.sam import SAM,visualize_image
import requests
import base64
import io
import cv2
import time
import torch
import torch.nn as nn
import gc
#from scipy.spatial.transform import Rotation as R



class semantic_keypoint:

    def __init__(self):

        self.detector = OWLViT()
        self.sam = SAM() 
        print("Foundation model loaded")

        self.img_size = 480
        background_image_path = "/home/adacomp/CLASP/semantic_keypoints/prototype/background.png"
        self.background = resize(Image.open(background_image_path).convert('RGB'), target_res=self.img_size, resize=True, to_pil=True)
        print("background loaded")
        self.prototype_image_root = "/home/adacomp/CLASP/semantic_keypoints/prototype"
    
    def update_observation(self, color_image):
        self.color_image = color_image
        self.raw_obs_shape = color_image.shape[:2]
        img = Image.fromarray(color_image)
        self.obs_mask, _ = self.detect_mask(img)
        print("update observation and mask")
    
    def process_rgb(self, visualize=False):
        obs_image = copy.deepcopy(self.color_image)
        obs_mask = copy.deepcopy(self.obs_mask)

        img = Image.fromarray(obs_image)

        moments = cv2.moments(obs_mask.astype(np.uint8))
        self.center_x = int(moments["m10"] / moments["m00"])
        self.center_y = int(moments["m01"] / moments["m00"])

        img_rotate = img.rotate(self.angle_rotate, expand=True)
        self.img_rotate = img_rotate

        mask_pil = Image.fromarray(obs_mask.astype(np.uint8))
        mask_rotate = mask_pil.rotate(self.angle_rotate, expand=True)
        print("rotate process")

        img_processed, mask_processed, self.transform_params = center_object(np.array(img_rotate), np.array(mask_rotate), self.background)
        print("center process")

        if visualize:
            visualize_image(img_processed, masks=[mask_processed], show=True, return_img=False) 

        return img_processed, mask_processed
    
    def process_rgb_without_rotation(self, visualize=False):
        obs_image = copy.deepcopy(self.color_image)
        obs_mask = copy.deepcopy(self.obs_mask)

        img_processed, mask_processed, _ = center_object(obs_image, obs_mask, self.background)
        print("center process")

        if visualize:
            visualize_image(img_processed, masks=[mask_processed], show=True, return_img=False) 

        return img_processed, mask_processed
    

    def detect_mask(self,obs_image, visualize=False):
        detected_objects = self.detector.detect_objects(
                image=obs_image,
                text_queries= ['cloth', 'trousers', 'skirt', 'tshirt', 'clothes', 'shirt'],   #'white object'], #['cloth', 'trousers', 'skirt', 'tshirt', ],
                bbox_score_top_k=25,
                bbox_conf_threshold=0.05)

        best_score = 0
        best_boxes = None
        for det in detected_objects:
            if det["score"] > best_score:
                best_score = det["score"]
                best_boxes = det['bbox']
        if best_boxes is None:
            print("No clothes detected")
            exit(0)
            
        masks = self.sam.segment_by_bboxes(image=obs_image, bboxes=[best_boxes])
        if visualize:
            visualize_image(obs_image, masks=[mask["segmentation"] for mask in masks], show=True, return_img=True)  

        return masks[0]["segmentation"], best_boxes  
    
    def detect_receptacle(self, obs_image, visualize=False): 
        detected_objects = self.detector.detect_objects(
        image=obs_image,
        text_queries= ['box', 'hanger'],   #'white object'], #['cloth', 'trousers', 'skirt', 'tshirt', ],
        bbox_score_top_k=25,
        bbox_conf_threshold=0.10)

        best_score = 0
        best_boxes = None
        for det in detected_objects:
            if det["score"] > best_score:
                best_score = det["score"]
                best_boxes = det['bbox']
        if best_boxes is None:
            print("No receptable detected")
            exit(0)

        masks = self.sam.segment_by_bboxes(image=obs_image, bboxes=[best_boxes])
        center_x, center_y, theta_rad = self.get_object_orientation(masks[0]["segmentation"])
        if visualize:
            visualize_image(obs_image, masks=[mask["segmentation"] for mask in masks], show=True, return_img=True)  
        return [center_x, center_y, theta_rad]
    
    def get_fuse_features(self,image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG") 

        img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = requests.post("http://crane1.ddns.comp.nus.edu.sg:4052/process_image", json={"image": img})

        response_json = response.json()
        fuse_features_bytes = base64.b64decode(response_json['features'])
        fuse_featuremap = np.load(io.BytesIO(fuse_features_bytes))
        dino_bytes = base64.b64decode(response_json['features_dino'])
        dino_featuremap = np.load(io.BytesIO(dino_bytes))
        return fuse_featuremap, dino_featuremap
    
    def get_dino_features(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG") 

        img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = requests.post("http://crane1.ddns.comp.nus.edu.sg:4052/process_image_dino", json={"image": img})

        response_json = response.json()
        dino_bytes = base64.b64decode(response_json['features_dino'])
        dino_featuremap = np.load(io.BytesIO(dino_bytes))
        return dino_featuremap
    
    def clothes_classification(self, num_patches=60):
        image_processed, mask_image_processed = self.process_rgb_without_rotation(visualize=False)
        obs_image = resize(Image.fromarray(image_processed), target_res=self.img_size, resize=True, to_pil=True)
        mask_image_processed = cv2.resize(mask_image_processed, (self.img_size,self.img_size), interpolation=cv2.INTER_NEAREST)
        
        obs_mask_pil = Image.fromarray((mask_image_processed > 0).astype(np.uint8))
        obs_mask_resized = obs_mask_pil.resize((num_patches, num_patches), Image.NEAREST)
        obs_mask_resized_np = np.array(obs_mask_resized).astype(np.uint8)
        
        query_dino_feature = self.get_dino_features(obs_image)
        query_ft = torch.tensor(query_dino_feature, dtype=torch.float32,device='cuda')

        best_score = float("-inf")
        best_prototype = None

        for fname in os.listdir(self.prototype_image_root):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")) or "m." in fname:
                continue  # skip masks or non-image files

            base_name = os.path.splitext(fname)[0]
            mask_name = base_name + ".npy"

            mask_path = os.path.join(self.prototype_image_root, mask_name)
            if not os.path.exists(mask_path):
                continue

            prototype_img_path = os.path.join(self.prototype_image_root, fname)
            prototype_imgage = Image.open(prototype_img_path).convert("RGB")
            prototype_dino_feature = self.get_dino_features(prototype_imgage)
            prototype_ft = torch.tensor(prototype_dino_feature, dtype=torch.float32,device='cuda')
            
            mask_np = np.load(mask_path)
            mask_img = Image.fromarray(mask_np.astype(np.uint8))
            mask_img_resized = mask_img.resize((num_patches, num_patches), Image.NEAREST)
            resized_mask_np = np.array(mask_img_resized).astype(np.uint8)
            ims_score = dino_simularity_with_bbs(query_ft, prototype_ft, obs_mask_resized_np, resized_mask_np)
            
            if ims_score > best_score:
                best_score = ims_score
                best_prototype = fname
        
        print("Best prototype found:", best_prototype, "with score:", best_score)
        return best_prototype, best_score

    def pose_matching_with_dino(self, num_patches=60):
        from torchvision.transforms import functional as TF

        # Resize source mask
        prototype_mask_pil = Image.fromarray(self.prototype_mask.astype(np.uint8))
        prototype_mask_resized = prototype_mask_pil.resize((num_patches, num_patches), Image.NEAREST)
        prototype_mask_np = np.array(prototype_mask_resized).astype(np.uint8)

        image_processed, mask_image_processed = self.process_rgb_without_rotation(visualize=False)
        obs_image = resize(Image.fromarray(image_processed), target_res=self.img_size, resize=True, to_pil=True)
        mask_image_processed = cv2.resize(mask_image_processed, (self.img_size,self.img_size), interpolation=cv2.INTER_NEAREST)
        
        obs_mask_pil = Image.fromarray((mask_image_processed > 0).astype(np.uint8))
        obs_mask_resized = obs_mask_pil.resize((num_patches, num_patches), Image.NEAREST)
        obs_mask_resized_np = np.array(obs_mask_resized).astype(np.uint8)

        best_score = float("-inf")
        best_angle = 0

        for angle_idx in range(12):
            angle = angle_idx * 360 / 12  # i.e., 0°, 30°, ..., 330°

            # Rotate query image and mask
            rotated_query_image = TF.rotate(obs_image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            rotated_query_mask = TF.rotate(Image.fromarray(obs_mask_resized_np), angle, interpolation=TF.InterpolationMode.NEAREST)

            # Resize and extract descriptors
            rotated_dino_feature = self.get_dino_features(rotated_query_image)
            rotated_mask_resized = rotated_query_mask.resize((num_patches, num_patches), Image.NEAREST)
            rotated_mask_np = np.array(rotated_mask_resized).astype(np.uint8)

            rotated_dino_feature_tensor = torch.tensor(rotated_dino_feature, dtype=torch.float32, device='cuda')
            prototype_dino_feature_tensor = torch.tensor(self.prototype_dino_features, dtype=torch.float32, device='cuda')

            # Compute similarity
            score = dino_simularity_with_bbs(rotated_dino_feature_tensor, prototype_dino_feature_tensor, rotated_mask_np, prototype_mask_np)

            if score > best_score:
                best_score = score
                best_angle = angle

        return int(best_angle), best_score        
    
    def load_prototype_image(self, clothes_category=None):   
        if clothes_category is not None:
            self.clothes_category = clothes_category
        else:
            self.clothes_category, _ = self.clothes_classification()

        prototype_image_dir = os.path.join(self.prototype_image_root, self.clothes_category.rsplit('.', 1)[0] + '.png')
        prototype_keypoint_dir = os.path.join(self.prototype_image_root, self.clothes_category.rsplit('.', 1)[0] + '.json')

        self.prototype_image = resize(Image.open(prototype_image_dir).convert('RGB'), target_res=self.img_size, resize=True, to_pil=True)
        self.prototype_features, _ = self.get_fuse_features(self.prototype_image)
        self.prototype_dino_features = self.get_dino_features(self.prototype_image)
        self.prototype_mask = np.load(os.path.join(self.prototype_image_root, self.clothes_category.rsplit('.', 1)[0] + '.npy'))
        
        print("prototype features loaded")  
        self.center_index = 1000
        with open(prototype_keypoint_dir) as f:
            prototype_kp_template_dic = json.load(f)
            prototype_kp_values =prototype_kp_template_dic.values()
            self.prototype_kp_descriptions = list(prototype_kp_template_dic.keys())
            self.prototype_kp_position = list(prototype_kp_values)
       
            if "center" in self.prototype_kp_descriptions:
                 self.center_index = self.prototype_kp_descriptions.index("center")
        
        print("prototype loaded")
    
    def pose_align(self, with_dino=False):
        if with_dino:
            self.angle_rotate,_ = self.pose_matching_with_dino()
        else:
            self.angle_rotate = mask_pose_alignment(self.prototype_mask, self.obs_mask)
        # self.angle_rotate = 0
        print("pose:" +str(self.angle_rotate)+ " degree")
    
    def pixel_backtrack(self, kp):  
        height_diff = self.transform_params["height_diff"]
        width_diff = self.transform_params["width_diff"]
        original_size = self.transform_params["original_size"]
        new_size = self.transform_params["new_size"]
        x = int(kp[1]*new_size/self.img_size)
        y = int(kp[0]*new_size/self.img_size)
        x = x-width_diff
        y = y-height_diff
        #mapping back to processd before

        # Shift point to origin
        x_shifted = x - original_size[1]/ 2
        y_shifted = y - original_size[0] / 2
            
        # Apply rotation formula
        angle_rad = self.angle_rotate*np.pi/180
        new_x = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
        new_y = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
            
        # Shift back to original center
        new_x += self.raw_obs_shape[1]/2
        new_y += self.raw_obs_shape[0]/2

        #mapping back to rotation bafore
        return [int(new_x), int(new_y)]

    
    def get_coarse_region_with_vlm(self, image, mask, kp):
        def find_candidate_region(res, mask):
            kp_region_name = None
            region_valid = False
            region_mapping = {"a3": (0, 0), "b3": (0, 1), "c3": (0, 2),
                            "a2": (1, 0),  "b2": (1, 1), "c2": (1, 2),
                            "a1": (2, 0),  "b1": (2, 1), "c1": (2, 2),}
            # Define the grid layout (3x3)
            height, width = mask.shape
            grid_rows, grid_cols = 3, 3
            cell_height, cell_width = height // grid_rows, width // grid_cols
            # Create an empty mask
            region_mask = np.zeros((height, width), dtype=np.uint8)
            for k in region_mapping.keys():
                if k in res:
                    kp_region_name = k
                    break

            if kp_region_name is not None:
                row, col = region_mapping[kp_region_name]
                x_start, y_start = col * cell_width, row * cell_height
                x_end, y_end = x_start + cell_width, y_start + cell_height
                region_mask[y_start:y_end, x_start:x_end] = 1
                region_mask = cv2.bitwise_and(region_mask, mask.astype(np.uint8))
                if np.sum(region_mask) > 0:
                    region_valid = True                    
            return region_valid, region_mask
        
        im1_with_grid = annotate_grid_with_kp(self.prototype_image, [3,3],kp)
        im2_with_grid = annotate_grid(image, [3,3])
        valid_region = False
        query_times = 0
        res = ""
        prompts = load_prompts()
        prompt_kp = prompts['global_matching']

        while not valid_region and query_times < 3:
            res = request_gpt("",[im1_with_grid,im2_with_grid], prompt_kp)
            valid_region, region_mask = find_candidate_region(res, mask)
            query_times += 1
            time.sleep(1)
        
        if not valid_region:
            print("No valid region found")
            region_mask = mask.astype(np.uint8)
        return region_mask
    
    def get_coarse_region_with_mask(self, mask, kp):
        def generate_mask(image_shape, pixel_location):
            height, width = image_shape
            row, col = pixel_location
            
            grid_height, grid_width = height // 2, width // 2
            row_grid = row // grid_height  # 0 or 1
            col_grid = col // grid_width   # 0 or 1
            
            mask = np.zeros((2, 2), dtype=np.uint8)
            mask[row_grid, col_grid] = 1  # Set the appropriate grid to 1
            expanded_mask = np.zeros(image_shape, dtype=np.uint8)
            for i in range(2):
                for j in range(2):
                    expanded_mask[i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width] = mask[i, j]
            
            return expanded_mask
        
        def generate_mask_center(image_shape, pixel_location):
            height, width = image_shape
            row, col = pixel_location
            
            # Define box size: 60% of the image dimensions
            box_height = int(0.6 * height)
            box_width = int(0.6 * width)
            
            # Calculate box boundaries centered at the keypoint
            top = max(0, row - box_height // 2)
            bottom = min(height, row + box_height // 2)
            left = max(0, col - box_width // 2)
            right = min(width, col + box_width // 2)
            
            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[top:bottom, left:right] = 1
            
            return mask
        
        region_mask = generate_mask_center((self.img_size, self.img_size),(kp[1], kp[0]))
        region_mask = cv2.bitwise_and(region_mask, mask.astype(np.uint8))
        valid_region = (np.sum(region_mask) > 0)

        if not valid_region:
            print("No valid region found")
            region_mask = np.ones((self.img_size, self.img_size), dtype=np.uint8)
        
        return region_mask
        
    def get_fine_kepoint(self, image_feature, global_region_mask, kp):   
        src_ft = torch.tensor(self.prototype_features, dtype=torch.float32,device='cuda')
        trg_ft = torch.tensor(image_feature, dtype=torch.float32,device='cuda') 

        y, x = int(kp[1]), int(kp[0])
        num_channel = src_ft.size(1)
        src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
        src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

        del src_ft
        gc.collect()
        torch.cuda.empty_cache()

        trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(trg_ft)
        cos = nn.CosineSimilarity(dim=1)
        raw_heatmap = cos(src_vec, trg_ft).cpu().numpy()[0] 

        del trg_ft
        gc.collect()
        torch.cuda.empty_cache()

        heatmap = copy.deepcopy(raw_heatmap) 
        heatmap = raw_heatmap * global_region_mask
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]

        match_point = np.where(heatmap == np.max(heatmap))
        match_point_pixel = [match_point[0][0], match_point[1][0]]  
        macth_point_pixel_original = self.pixel_backtrack(match_point_pixel)
        return macth_point_pixel_original, raw_heatmap  
    
    def keypoints_extraction(self, with_vlm=False):
        image_processed, mask_image_processed = self.process_rgb(visualize=False)
        obs_image = resize(Image.fromarray(image_processed), target_res=self.img_size, resize=True, to_pil=True)
        mask_image_processed = cv2.resize(mask_image_processed, (self.img_size,self.img_size), interpolation=cv2.INTER_NEAREST)
        obs_mask = (mask_image_processed > 0).astype(np.uint8)
        obs_features, _ = self.get_fuse_features(obs_image)
        print("obs features loaded")
        detect_kp_list = []
        for kp_index in range(len(self.prototype_kp_descriptions)):
            kp_description = self.prototype_kp_descriptions[kp_index]
            if kp_description == "center":
                detect_kp = [self.center_x, self.center_y]
            else:
                if with_vlm:
                    global_region_mask = self.get_coarse_region_with_vlm(obs_image, obs_mask, self.prototype_kp_position[kp_index])
                else:
                    global_region_mask = self.get_coarse_region_with_mask(obs_mask, self.prototype_kp_position[kp_index])
                detect_kp, _ = self.get_fine_kepoint(obs_features, global_region_mask, self.prototype_kp_position[kp_index])
            detect_kp_list.append(detect_kp)

        return detect_kp_list
    
    def draw_keypoint(self, detect_kp_list, visualize):
        with_descriptor = True
        canvas = copy.deepcopy(self.color_image)
        colors_list = [
        (100, 0, 0),    # Dark Blue
        (0, 100, 0),    # Dark Green
        (0, 0, 100),    # Dark Red
        (100, 100, 0),  # Dark Cyan
        (100, 0, 100),  # Dark Magenta
        (0, 100, 100),  # Dark Yellow
        (100, 50, 0),   # Dark Orange
        (50, 0, 100),   # Dark Purple
        (100, 0, 50),   # Dark Pink
        ]

        text_color = (255, 255, 255)
        for i in range(len(detect_kp_list)):
            x, y = detect_kp_list[i]
            cv2.circle(canvas, (int(x), int(y)), 10, colors_list[i], -1) 
            if with_descriptor:
                cv2.putText(canvas, self.prototype_kp_descriptions[i], (int(x) + 12, int(y) - 12),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

        if visualize:   
            plt.imshow(canvas)
            plt.axis('off')
            plt.show()

        return canvas



if __name__ == "__main__":
    semantic_keypoint_instance = semantic_keypoint()
    image_dir = "/home/adacomp/CLASP/test.png"
    raw_image = cv2.imread(image_dir)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    semantic_keypoint_instance.update_observation(raw_image)
    semantic_keypoint_instance.load_prototype_image()
    semantic_keypoint_instance.pose_align(with_dino=True)
    detect_kp_positions = semantic_keypoint_instance.keypoints_extraction()
    semantic_keypoint_instance.draw_keypoint(detect_kp_positions, visualize=True)

