import numpy as np
import cv2 as cv

def detect_direction_changes(sampled_points, angle_threshold=45):
    point_len = len(sampled_points)


    sampled_points_added = np.concatenate(([sampled_points[point_len-1]], sampled_points, [sampled_points[0]]), axis=0)


    direction_change_points = []
    for i in range(1, len(sampled_points_added) - 1):
        # Get three consecutive points
        p1 = sampled_points_added[i - 1]
        p2 = sampled_points_added[i]
        p3 = sampled_points_added[i + 1]

        # Compute vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Compute the angle between the two vectors
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))  # Use dot product to find the angle

        # If the angle is greater than the threshold, mark it as a direction change point
        if angle > angle_threshold:
            direction_change_points.append(p2)
    
    return np.array(direction_change_points, dtype=np.int32)


def detect_robust_direction_changes(sampled_points, window_size=5, angle_threshold=45):
    direction_change_points = []
    # print("sampled_points", sampled_points.shape)
    point_len = len(sampled_points)

    sampled_points_added = np.concatenate((sampled_points[point_len-window_size:point_len], sampled_points, sampled_points[:window_size]), axis=0)

    for i in range(window_size, len(sampled_points_added) - window_size):
        # Get the window of points before and after the current point
        before_window = sampled_points_added[i - window_size:i]
        after_window = sampled_points_added[i + 1:i + 1 + window_size]

        # Calculate average direction vectors
        before_vector = np.mean(np.diff(before_window, axis=0), axis=0)
        after_vector = np.mean(np.diff(after_window, axis=0), axis=0)

        # Normalize vectors
        before_vector = before_vector / np.linalg.norm(before_vector)
        after_vector = after_vector / np.linalg.norm(after_vector)

        # Compute the angle between the average vectors
        angle = np.degrees(np.arccos(np.clip(np.dot(before_vector, after_vector), -1.0, 1.0)))

        # If the angle exceeds the threshold, consider it a direction change
        if angle > angle_threshold:
            direction_change_points.append(sampled_points_added[i])

    return np.array(direction_change_points, dtype=np.int32)

def remove_overlapping_keypoints(keypoints, min_distance=20):
    unique_keypoints = []
    
    for point in keypoints:
        if len(unique_keypoints) == 0:
            unique_keypoints.append(point)
        else:
            distances = [np.linalg.norm(np.array(point) - np.array(kp)) for kp in unique_keypoints]
            if all(dist > min_distance for dist in distances):
                unique_keypoints.append(point)
    
    return np.array(unique_keypoints, dtype=np.int32)


# Function to detect endpoints
def find_endpoints(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Filter the skeleton to find points with exactly 2 neighbors (endpoints)
    neighbors = cv.filter2D(skeleton.astype(np.uint8), -1, kernel) 
    endpoints = np.where(neighbors == 11)
    return endpoints

# Function to detect branch points
def find_branchpoints(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Filter the skeleton to find points with more than 2 neighbors (branch points)
    neighbors = cv.filter2D(skeleton.astype(np.uint8), -1, kernel)
    branchpoints = np.where(neighbors > 12)
    return branchpoints

def fps(points, n_samples):
    """
    points: [N, 2] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype="int")  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float("inf")  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left]) ** 2
        ).sum(
            -1
        )  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(
            dist_to_last_added_point, dists[points_left]
        )  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]