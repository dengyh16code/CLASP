import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed

import argparse
import base64
import io
import os
import json
import numpy as np
import torch

from PIL import Image

from flask import Flask, request, jsonify

set_seed(42)
num_patches = 60
sd_model = sd_aug = extractor_vit = None
aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
aggre_net.load_pretrained_weights(torch.load('/data/home/anxing/GeoAware-SC/results_spair/best_856.PTH'))
sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
print("Model loaded")

def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    # extract stable diffusion features
    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    # extract dinov2 features
    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    
    desc = aggre_net(desc_gathered) # 1, 768, 60, 60
    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc, features_dino


def get_processed_features_dino(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    # extract dinov2 features
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        dino_path = f"{feature_base}_dino.pt"
        
        if os.path.exists(dino_path):
            features_dino = torch.load(dino_path)
        else:
            if img is None: img = Image.open(img_path).convert('RGB')
            img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
            img_batch = extractor_vit.preprocess_pil(img_dino_input)
            features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)
    else:
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    # normalize the descriptors
    norms_desc = torch.linalg.norm(features_dino, dim=1, keepdim=True)
    features_dino = features_dino / (norms_desc + 1e-8)
    return features_dino

# img_size = 480
# img1_path = '/data/home/anxing/GeoAware-SC/data/images/unicorn.jpg' # path to the source image
# img1 = resize(Image.open(img1_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True)

# img2_path = '/data/home/anxing/GeoAware-SC/data/images/antelope.jpg' # path to the target image
# img2 = resize(Image.open(img2_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True)

# # visualize the two images in the same row
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# for a in ax: a.axis('off')
# ax[0].imshow(img1)
# ax[0].set_title('source image')
# ax[1].imshow(img2)
# ax[1].set_title('target image')
# plt.show()

# feat1 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img1)
# feat2 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=img2)

# Flask app
app = Flask(__name__)

def decode_base64(base64_image: str):
    image_data = base64.b64decode(base64_image)
    image_data = io.BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_base64(data['image'])
        
        # Get features
        features, features_dino = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=image)
        
        # Convert to numpy arrays
        features_np = features.cpu().detach().numpy()
        features_dino_np = features_dino.cpu().detach().numpy()
        
        # Convert numpy arrays to compressed base64 strings
        features_bytes = io.BytesIO()
        np.save(features_bytes, features_np)
        features_base64 = base64.b64encode(features_bytes.getvalue()).decode('utf-8')
        
        features_dino_bytes = io.BytesIO()
        np.save(features_dino_bytes, features_dino_np)
        features_dino_base64 = base64.b64encode(features_dino_bytes.getvalue()).decode('utf-8')
        
        return jsonify({
            'features': features_base64,
            'features_dino': features_dino_base64,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_image_dino', methods=['POST'])
def process_image_dino():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_base64(data['image'])
        
        # Get features
        features_dino = get_processed_features_dino(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=image)
        
        # Convert to numpy arrays
        features_dino_np = features_dino.cpu().detach().numpy()
        
        # Convert numpy arrays to compressed base64 strings
        features_dino_bytes = io.BytesIO()
        np.save(features_dino_bytes, features_dino_np)
        features_dino_base64 = base64.b64encode(features_dino_bytes.getvalue()).decode('utf-8')
        
        return jsonify({
            'features_dino': features_dino_base64,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

##### to be completed
#####


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geo Aware SC Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=4002, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)
