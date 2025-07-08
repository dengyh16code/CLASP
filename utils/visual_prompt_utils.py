from string import ascii_lowercase
import os
import numpy as np
import sys
sys.path.append("/home/adacomp/CLASP")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from PIL import Image
import io
import traceback
import json
from utils.gpt_utils import request_gpt
from utils.gpt_utils import request_gpt_incontext

from openai import OpenAI
client = OpenAI()


# --------------------------------
# Proposal and Selection
# --------------------------------


def parse_json_string(res, verbose=False):
    if '```' in res:
        try:
            res_clean = res

            if '```json' in res:
                res_clean = res_clean.split('```')[1].split('json')[1]
            elif '```JSON' in res:
                res_clean = res_clean.split('```')[1].split('JSON')[1]
            elif '```' in res:
                res_clean = res_clean.split('```')[1]
            else:
                print('Invalid response: ')
                print(res)

        except Exception:
            print(traceback.format_exc())
            print('Invalid response: ')
            print(res)
            return None
    else:
        res_clean = res

    try:
        res_filtered = remove_trailing_comments(res_clean)
        object_info = json.loads(res_filtered)

        # if verbose:
        #     print_object_info(object_info)

        return object_info

    except Exception:
        print(traceback.format_exc())
        print('The original response: ')
        print(res)
        print('Invalid cleaned response: ')
        print(res_clean)
        return None


def remove_trailing_comments(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')

    # Process each line to remove comments
    processed_lines = []
    for line in lines:
        comment_index = line.find('//')
        if comment_index != -1:
            # Remove the comment
            line = line[:comment_index]
        processed_lines.append(line.strip())

    # Join the processed lines back into a single string
    return """{}""".format('\n'.join(processed_lines))


def request_plan(
        task_instruction,
        obs_image,
        plan_with_obs_image,
        prompts,
        example_images=None,
        example_responses=None,
        debug=False):
    """Decompose the task into subtasks.
    """
    text_requests = [f"Task: {task_instruction}"]

    if plan_with_obs_image:
        image_requests = [obs_image]
    else:
        image_requests = []

    # Generate the subtask plan.
    if (example_images is not None and
            example_responses is not None):
        assert len(example_images) == len(example_responses)
        res = request_gpt_incontext(
            text_requests,
            image_requests,
            prompts['propose_subtasks'],
            example_images=example_images,
            example_responses=example_responses)
    else:
        res = request_gpt(
            text_requests,
            image_requests,
            prompts['propose_subtasks'])
    
    # Merge picking and placing into higher-evel subtasks.
    res_filtered = request_gpt(
        text_requests + [res],
        image_requests,
        prompts['filter_subtasks'])

    if debug:
        print('--------------------------------')
        print('| Generated plan.')
        print('--------------------------------')
        print(res)

        print('--------------------------------')
        print('| Filtered plan.')
        print('--------------------------------')
        print(res_filtered)

    plan_info = parse_json_string(res_filtered)

    return plan_info

def request_plan_with_kp(
        task_instruction,
        semantic_keypoints,
        obs_image,
        plan_with_obs_image,
        prompts,
        example_images=None,
        example_responses=None):
    """Decompose the task into subtasks.
    """
    text_requests = [f"task: {task_instruction}, semantic keypoints: {semantic_keypoints}"]

    if plan_with_obs_image:
        image_requests = [obs_image]
    else:
        image_requests = []

    # Generate the subtask plan.
    if (example_images is not None and
            example_responses is not None):
        assert len(example_images) == len(example_responses)
        res = request_gpt_incontext(
            text_requests,
            image_requests,
            prompts['propose_subtasks'],
            example_images=example_images,
            example_responses=example_responses)
    else:
        res = request_gpt(
            text_requests,
            image_requests,
            prompts['propose_subtasks'])

    return res

def request_plan_without_kp(
        task_instruction,
        obs_image,
        plan_with_obs_image,
        prompts,
        example_images=None,
        example_responses=None):
    """Decompose the task into subtasks.
    """
    text_requests = [f"task: {task_instruction}"]

    if plan_with_obs_image:
        image_requests = [obs_image]
    else:
        image_requests = []

    # Generate the subtask plan.
    if (example_images is not None and
            example_responses is not None):
        assert len(example_images) == len(example_responses)
        res = request_gpt_incontext(
            text_requests,
            image_requests,
            prompts['propose_subtasks_nokp'],
            example_images=example_images,
            example_responses=example_responses)
    else:
        res = request_gpt(
            text_requests,
            image_requests,
            prompts['propose_subtasks_nokp'])

    return res


def request_motion(  # NOQA
        subtask,
        obs_image_reshaped,
        annotated_image,
        candidate_keypoints,
        waypoint_grid_size,
        prompts,
        debug=False,
        example_images=None,
        example_responses=None,
        loaded_context=None,
        use_center=False,
        log_dir=None,
        suffix='',
        add_caption=True,
):
    """Generate the visual marks that specify the motion.
    """
    annotation_size = obs_image_reshaped.size[:2]
    # print('annotation_size (request_motion)', annotation_size)

    text_requests = [f"Task: {subtask}"]

    if loaded_context is not None:
        context = loaded_context
    else:
        context = None
        while context is None:
            if (example_images is not None and
                    example_responses is not None):
                print('example_respnses:')
                print(example_responses)
                assert len(example_images) == len(example_responses)
                res = request_gpt_incontext(
                    text_requests,
                    [annotated_image],
                    prompts['select_motion'],
                    example_images=example_images,
                    example_responses=example_responses)
            else:
                res = request_gpt(
                    text_requests,
                    [annotated_image],
                    prompts['select_motion'])

            if debug:
                print('--------------------------------')
                print('| Selected motion.')
                print('--------------------------------')
                print(res)
            context = parse_json_string(res)

    context_json = context
    grasp_keypoint = None
    val = context['grasp_keypoint']
    if val != '':
        if use_center:
            idx = 0
        else:
            idx = int(val[1:]) - 1
        grasp_keypoint = candidate_keypoints['grasped'][idx]

    function_keypoint = None
    val = context['function_keypoint']
    if val != '':
        idx = int(val[1:]) - 1
        function_keypoint = candidate_keypoints['grasped'][idx]

    target_keypoint = None
    val = context['target_keypoint']
    if val != '':
        if use_center:
            idx = 0
        else:
            idx = int(val[1:]) - 1
        target_keypoint = candidate_keypoints['unattached'][idx]
    else:
        # A patch for the missing target keypoint issue during pick-and-place.
        pass
        # if (
        #         subtask['object_grasped'] is not None and
        #         subtask['object_unattached'] is not None
        # ):
        #     target_keypoint = candidate_keypoints['unattached'][0]

    pre_contact_tile = None
    pre_contact_waypoint = None
    val = context['pre_contact_tile']
    if val != '':
        pre_contact_tile = [
            waypoint_grid_size[1] - int(val[1]),
            ascii_lowercase.index(val[0]),
        ]
        pre_contact_waypoint = sample_waypoint_in_tile(
            annotation_size, waypoint_grid_size, pre_contact_tile)

    post_contact_tile = None
    post_contact_waypoint = None
    val = context['post_contact_tile']
    if val != '':
        post_contact_tile = [
            waypoint_grid_size[1] - int(val[1]),
            ascii_lowercase.index(val[0]),
        ]
        post_contact_waypoint = sample_waypoint_in_tile(
            annotation_size, waypoint_grid_size, post_contact_tile)

    pre_contact_height = context['pre_contact_height']
    post_contact_height = context['post_contact_height']
    target_angle = context['target_angle']

    # if target_keypoint is None:
    #     pre_contact_tile = None
    #     pre_contact_waypoint = None
    #     post_contact_tile = None
    #     post_contact_waypoint = None

    if function_keypoint is None:
        target_angle = 'forward'

    context = dict(
        keypoints_2d=dict(
            grasp=grasp_keypoint,
            function=function_keypoint,
            target=target_keypoint,
        ),
        waypoints_2d=dict(
            pre_contact=[pre_contact_waypoint],
            post_contact=[post_contact_waypoint],
        ),
        target_euler=target_angle,
        pre_contact_height=pre_contact_height,
        post_contact_height=post_contact_height,
    )

    if debug:
        # if True:
        log_img = annotate_motion(obs_image_reshaped, context,
                                  add_caption=add_caption)
        if log_dir is not None:
            log_img.save(os.path.join(log_dir, f'motion{suffix}.png'))
            out_file = open(os.path.join(log_dir, f'context{suffix}.json'), 'w')
            json.dump(context_json, out_file)

        return context, context_json, log_img
    else:
        return context, context_json, None




def sample_waypoint_in_tile(image_size, grid_size, tile):
    w, h = image_size
    h_min = h * (tile[0]) / grid_size[0]
    h_max = h * (tile[0] + 1) / grid_size[0]
    w_min = w * (tile[1]) / grid_size[1]
    w_max = w * (tile[1] + 1) / grid_size[1]
    y = np.random.uniform(h_min, h_max)
    x = np.random.uniform(w_min, w_max)
    return np.array([x, y])


# --------------------------------
# Annotation and Visualization
# --------------------------------


def plot_keypoints(
        ax, image_size, keypoints, color, prefix='', annotate_index=True,
        add_caption=True):
    if keypoints is None:
        return

    (h, w) = image_size
    for i, keypoint in enumerate(keypoints):
        if keypoint is None:
            continue

        ax.plot(
            keypoint[0], keypoint[1],
            color=color, alpha=0.4,
            marker='o', markersize=12,
            markeredgewidth=3, markeredgecolor='gray',
        )

        if add_caption:
            text = ''
            if annotate_index:
                text = text + str(i + 1)
            text = prefix + text

            xytext = (
                min(max(0, keypoint[0]), h ),
                min(max(0, keypoint[1]), w ),
            )

            ax.annotate(text, keypoint, xytext, size=20, color='black')


# def annotate_candidate_keypoints_bef(
#         image,
#         candidate_keypoints,
# ):
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(image)
#     ax.axis('off')

#     image_size = image.size[:2]
#     print('image_size (annotate_candidate_keypoints)', image_size)

#     if candidate_keypoints['grasped'] is not None:
#         plot_keypoints(
#             ax, image_size, candidate_keypoints['grasped'], 'r', prefix='P')

#     if candidate_keypoints['unattached'] is not None:
#         plot_keypoints(
#             ax, image_size, candidate_keypoints['unattached'], 'b', prefix='Q')

#     buf = io.BytesIO()
#     fig.savefig(buf, transparent=True, bbox_inches='tight',
#                 pad_inches=0, format='jpg')
#     buf.seek(0)
#     # close the figure to prevent it from being displayed
#     plt.close(fig)
#     return Image.open(buf)


def annotate_candidate_keypoints(
        image,
        candidate_keypoints,
        add_caption=True,
):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]
    print('image_size (annotate_candidate_keypoints)', image_size)

    plot_keypoints(ax, image_size, candidate_keypoints, 'b', add_caption=add_caption, prefix='P')

    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)

def convert_list_format(items):
    """
    Converts a list of strings into a new format with indexed prefixes (P1, P2, ...).
    
    Args:
        items (list): List of strings.
    
    Returns:
        list: Transformed list with prefixed indices.
    """
    return [f"P{i+1}-{item}" for i, item in enumerate(items)]




def annotate_grid_with_kp(image, grid_size, kp, with_grid = True):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]
    (w, h) = image_size
    
    if with_grid:
        for i in range(1, grid_size[0]):
            ax.hlines(h * i / grid_size[0], 0, w,
                    color='black', alpha=0.3, linewidth=1.5)

        for j in range(1, grid_size[1]):
            ax.vlines(w * j / grid_size[1], 0, h,
                    color='black', alpha=0.3, linewidth=1.5)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(str(i + 1),
    #                 [w * (i + 0.5) / grid_size[0], 0],
    #                 [w * (i + 0.5) / grid_size[0], -10],
    #                 size=12)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(ascii_lowercase[i],
    #                 [0, h * (i + 0.5) / grid_size[0]],
    #                 [-20, h * (i + 0.5) / grid_size[0]],

    #                 size=12)

    plot_keypoints(ax, image_size, [kp], 'r', add_caption=False)

    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)


def annotate_grid(image, grid_size):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]
    (w, h) = image_size

    for i in range(1, grid_size[0]):
        ax.hlines(h * i / grid_size[0], 0, w,
                  color='black', alpha=0.3, linewidth=1.5)

    for j in range(1, grid_size[1]):
        ax.vlines(w * j / grid_size[1], 0, h,
                  color='black', alpha=0.3, linewidth=1.5)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(str(i + 1),
    #                 [w * (i + 0.5) / grid_size[0], 0],
    #                 [w * (i + 0.5) / grid_size[0], -10],
    #                 size=12)

    # for i in range(0, grid_size[0]):
    #     ax.annotate(ascii_lowercase[i],
    #                 [0, h * (i + 0.5) / grid_size[0]],
    #                 [-20, h * (i + 0.5) / grid_size[0]],
    #                 size=12)

    for i in range(0, grid_size[0]):
        for j in range(0, grid_size[1]):
            ax.annotate(str(f"{ascii_lowercase[i]}{grid_size[1] - j}"),
                        [w * (i + 0.5) / grid_size[0], h *
                         (j + 0.5) / grid_size[1]],
                        [w * (i + 0.5) / grid_size[0], h *
                         (j + 0.5) / grid_size[1]],
                        size=15,
                        color='black')

    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)
    return Image.open(buf)


def annotate_visual_prompts(
        obs_image,
        candidate_keypoints,
        waypoint_grid_size,
        log_dir=None,
):
    """Annotate the visual prompts on the image.
    """
    annotated_image = annotate_candidate_keypoints(
        obs_image,
        candidate_keypoints,
    )
    if log_dir is not None:
        annotated_image.save(os.path.join(log_dir, 'keypoints.png'))
    annotated_image = annotate_grid(
        annotated_image,
        waypoint_grid_size,
    )
    if log_dir is not None:
        annotated_image.save(os.path.join(log_dir, 'grid.png'))
    else:
        plt.imshow(annotated_image)
        plt.show()
    return annotated_image


def plot_smooth_curve(ax, points):
    points = np.array(points)
    inds = np.argsort(points[:, 0])  # [::-1]
    points = points[inds]

    xs = []
    ys = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])

    xs = np.array(xs)
    ys = np.array(ys)

    # print(xs)
    # print(ys)
    # input()

    if len(xs) > 2:
        x_smooth = np.linspace(xs.min(), xs.max(), 100)
        spl = make_interp_spline(xs, ys, k=2)
        y_smooth = spl(x_smooth)
    else:
        x_smooth = xs
        y_smooth = ys

    ax.plot(
        x_smooth,
        y_smooth,
        linestyle=':',
        color='cyan',
        alpha=0.4,
        linewidth=3,
        zorder=0,
    )


def annotate_motion(image, context, log_dir=None, add_caption=True):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')

    image_size = image.size[:2]

    grasp_keypoint = context['keypoints_2d']['grasp']
    function_keypoint = context['keypoints_2d']['function']
    target_keypoint = context['keypoints_2d']['target']
    pre_contact_waypoint = context['waypoints_2d']['pre_contact'][0]
    post_contact_waypoint = context['waypoints_2d']['post_contact'][0]

    plot_keypoints(
        ax, image_size, [grasp_keypoint], 'r', 'grasp', False,
        add_caption=add_caption)
    plot_keypoints(
        ax, image_size, [function_keypoint], 'y', 'function', False,
        add_caption=add_caption)
    plot_keypoints(
        ax, image_size, [target_keypoint], 'b', 'target', False,
        add_caption=add_caption)
    plot_keypoints(
        ax, image_size, [pre_contact_waypoint], 'cyan', 'waypoint 0', False,
        add_caption=add_caption)
    plot_keypoints(
        ax, image_size, [post_contact_waypoint], 'cyan', 'waypoint 1', False,
        add_caption=add_caption)

    # if target_keypoint is not None and pre_contact_waypoint is not None:
    #     ax.plot(
    #         [pre_contact_waypoint[0], target_keypoint[0]],
    #         [pre_contact_waypoint[1], target_keypoint[1]],
    #         color='b', alpha=0.4,
    #     )

    # if target_keypoint is not None and post_contact_waypoint is not None:
    #     ax.plot(
    #         [target_keypoint[0], post_contact_waypoint[0]],
    #         [target_keypoint[1], post_contact_waypoint[1]],
    #         color='b', alpha=0.4,
    #     )

    waypoints = [
        point
        for point in [
            pre_contact_waypoint, target_keypoint, post_contact_waypoint]
        if point is not None]
    plot_smooth_curve(ax, waypoints)

    plt.show()
    buf = io.BytesIO()
    fig.savefig(buf, transparent=True, bbox_inches='tight',
                pad_inches=0, format='jpg')
    buf.seek(0)
    # close the figure to prevent it from being displayed
    plt.close(fig)

    return Image.open(buf)

def load_few_shot_examples(examples_dir):
    example_images = []
    example_tasks = []
    example_responses = []
    # Load data from folder
    file_numbers = [
        int(os.path.splitext(f)[0]) for f in os.listdir(examples_dir) if f.endswith((".json", ".png"))
    ]
    max_num = max(file_numbers) if file_numbers else -1  # Handle empty folder case

    # Load files in numerical order
    for i in range(max_num + 1):
        json_path = os.path.join(examples_dir, f"{i}.json")
        img_path = os.path.join(examples_dir, f"{i}.png")

        # Load JSON file
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                example_tasks.append(data["task"])
                response_diction = {"completed": data["completed"], "subtasks": data["subtasks"]}
                #example_responses.append({"completed": data["completed"], "subtasks": data["subtasks"]})

                example_response = json.dumps(response_diction, indent=4)
                example_response = f"""```json{example_response}```"""
                example_responses.append(example_response)

        # Load image
        if os.path.exists(img_path):
            example_images.append(Image.open(img_path))
    
    return example_images, example_tasks, example_responses
