import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def batch_cosine_similarity(feature_matrix1, feature_matrix2, batch_size=64):

    feature_matrix1 = feature_matrix1.view(768, -1).transpose(0, 1)  # [480*480, 768]
    feature_matrix2 = feature_matrix2.view(768, -1).transpose(0, 1)  # [480*480, 768]

    feature_matrix1_norm = F.normalize(feature_matrix1, dim=1)
    feature_matrix2_norm = F.normalize(feature_matrix2, dim=1)

    num_samples = feature_matrix1_norm.shape[0]
    cosine_similarity_matrix = torch.zeros((num_samples, num_samples), device=feature_matrix1_norm.device)

    for i in range(0, num_samples, batch_size):
        end_i = min(i + batch_size, num_samples)
        batch1 = feature_matrix1_norm[i:end_i]
        
        for j in range(0, num_samples, batch_size):
            end_j = min(j + batch_size, num_samples)
            batch2 = feature_matrix2_norm[j:end_j]
            
            cosine_similarity_matrix[i:end_i, j:end_j] = torch.mm(batch1, batch2.transpose(0, 1))

    return cosine_similarity_matrix

def dino_simularity_with_bbs(src_ft, trg_ft, mask1, mask2):
    avg_tensor1 = src_ft.mean(dim=1)
    avg_tensor2 = trg_ft.mean(dim=1)
    saliency_map1 = avg_tensor1.view(-1)
    saliency_map2 = avg_tensor2.view(-1)
    flat_mask_1 = mask1.flatten() 
    flat_mask_2 = mask2.flatten() 

    fg_mask1 = torch.from_numpy(flat_mask_1 > 0).cuda()
    fg_mask2 = torch.from_numpy(flat_mask_2 > 0).cuda()

    cosine_sim_matrix = batch_cosine_similarity(src_ft, trg_ft, batch_size=1024)#3600*3600
    cosine_sim_matrix = cosine_sim_matrix.unsqueeze(0).unsqueeze(0)
    
    num_patches1 = (60,60)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device='cuda')
    sim_1, nn_1 = torch.max(cosine_sim_matrix, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(cosine_sim_matrix, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0] #4015
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0] #3080
    bbs_mask = nn_2[nn_1] == image_idxs #4015

    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device='cuda')
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)#4015

    descriptors1 = src_ft.permute(0, 2, 3, 1)  # 1 x 60 x 60 x 768
    descriptors1 = descriptors1.reshape(1, 1, 3600, 768)
    descriptors2 = src_ft.permute(0, 2, 3, 1)  # 1 x 60 x 60 x 768
    descriptors2 = descriptors2.reshape(1, 1, 3600, 768)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].detach().cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].detach().cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(100, len(all_keys_together))  # if not enough pairs, show all found pairs.
    # n_clusters = min(20, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)  #10

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i
    
    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
    bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device='cuda')[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]

    # compute IMS (instance matching similarity) using two descriptors. the higher, the better
    ims = 0
    for i in range(len(img1_indices_to_show)):
        matching_dist = cosine_sim_matrix[0][0][img1_indices_to_show[i], img2_indices_to_show[i]]
        ims += matching_dist

    return ims


def center_object(img, mask, background, size_ratio=1.05):   
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Get bounding box coordinates
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    original_size =[img.shape[0],img.shape[1]]

    width = int(x2) - int(x1)
    height = int(y2) - int(y1)
    img_crop = img[int(y1):int(y2), int(x1):int(x2)]
    mask_crop = mask[int(y1):int(y2), int(x1):int(x2)]

    width_offset = int(width*size_ratio)
    height_offset = int(height*size_ratio)

    image_size = max(width_offset, height_offset)
    background_array = cv2.resize(np.array(background), (image_size, image_size))

    image_new = cv2.resize(np.array(background), (image_size, image_size))
    mask_new = np.zeros((image_size, image_size))
        
    start_height = int(image_size/2-height/2)
    end_height = start_height+height
    start_width = int(image_size/2-width/2)
    end_width = start_width+width
    
    #center
    mask_new[start_height:end_height, start_width:end_width] = mask_crop
    image_new[start_height:end_height, start_width:end_width, :] = img_crop
    
    #replace background
    image_processed = np.where(mask_new[..., None] == 1, image_new, background_array)
    
    transform_params = {"height_diff": start_height-int(y1), "width_diff": start_width-int(x1), "original_size": original_size, "new_size": image_size}

    return image_processed, mask_new, transform_params



def mask_pose_alignment(mask_ref, mask):
        mask_width = 200
        mask_height = 200
        static_mask_1 = Image.fromarray(mask_ref)
        static_mask_2 = Image.fromarray(mask)

        def preprocess_mask(mask,h, w):
            # Find the bounding box of the mask
            x, y, w_bbox, h_bbox = cv2.boundingRect(mask.astype(np.uint8))
            cropped = mask[y:y+h_bbox, x:x+w_bbox]

            # Resize the mask to the desired dimensions
            resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
            return resized
        
        def iou_mask(reference_mask, mask2):
            best_index = -1
            max_iou = 0
            for i, mask in enumerate(mask2):
                intersection = np.logical_and(reference_mask, mask)
                union = np.logical_or(reference_mask, mask)
                iou_score = np.sum(intersection) / np.sum(union)
                if iou_score > max_iou:
                    max_iou = iou_score
                    best_index = i

            return best_index, max_iou  

        mask1_processed = preprocess_mask(np.array(static_mask_1).astype(np.uint8), mask_height, mask_width)
        mask2_processed_list = []
        rotation_angle_list = list(range(0, 360, 15))
        for angle_rotate in rotation_angle_list:
            mask2_rotate = static_mask_2.rotate(angle_rotate, expand=True)
            mask2_processed = preprocess_mask(np.array(mask2_rotate).astype(np.uint8), mask_height, mask_width)
            mask2_processed_list.append(mask2_processed)
        
        best_index, _ = iou_mask(mask1_processed, mask2_processed_list)
        return rotation_angle_list[best_index]



def resize(img, target_res=224, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.BICUBIC) #Image.Resampling.LANCZOS
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.BICUBIC)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.BICUBIC)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.BICUBIC)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas
