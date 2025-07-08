import os
import base64
import requests
from io import BytesIO


# Get OpenAI API Key from environment variable

api_key = os.environ['OPENAI_API_KEY']
print("openai loaded")

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

DEFAULT_LLM_MODEL_NAME = 'gpt-4'
DEFAULT_VLM_MODEL_NAME = 'gpt-4o-2024-05-13'

def load_prompts():
    """Load prompts from files.
    """
    prompts = dict()
    prompt_dir = "/home/adacomp/CLASP/prompts"
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path[-4:] == '.txt':
            with open(path, 'r') as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value
    return prompts


def encode_image_from_file(image_path):
    # Function to encode the image
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def prepare_inputs(messages,
                   images,
                   meta_prompt,
                   model_name,
                   local_image):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)
    


    if not isinstance(images, list):
        images = [images]

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)
    

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800
    }

    return payload


def request_gpt(message,
                images,
                meta_prompt='',
                model_name=None,
                local_image=False):


    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_inputs(message,
                             images,
                             meta_prompt=meta_prompt,
                             model_name=model_name,
                             local_image=local_image)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception:
        print('\nInvalid response: ')
        print(response)
        print('\nInvalid response: ')
        print(response.json())
        exit()

    return res


def prepare_inputs_incontext(
        messages,
        images,
        meta_prompt,
        model_name,
        local_image,
        example_images,
        example_responses,
):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for example_image, example_response in zip(
            example_images, example_responses):
        if local_image:
            base64_image = encode_image_from_file(example_image)
        else:
            base64_image = encode_image_from_pil(example_image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

        content = {
            'type': 'text',
            'text': example_response,
        }
        user_content.append(content)

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800
    }

    return payload


def request_gpt_incontext(
        message,
        images,
        meta_prompt='',
        example_images=None,
        example_responses=None,
        model_name=None,
        local_image=False):

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_inputs_incontext(
        message,
        images,
        meta_prompt=meta_prompt,
        model_name=model_name,
        local_image=local_image,
        example_images=example_images,
        example_responses=example_responses)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception:
        print('\nInvalid response: ')
        print(response)
        print('\nInvalid response: ')
        print(response.json())
        exit()

    return res
