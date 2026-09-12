import os
import base64
import requests
from io import BytesIO


CLASP_HOME = os.environ['CLASP_HOME']

DEFAULT_LLM_MODEL_NAME = 'gpt-5.6'
DEFAULT_VLM_MODEL_NAME = 'gpt-5.6'


# The dictionary key is the name passed to request_gpt(). ``model_name`` is
# the name sent to the provider, which lets aliases such as ``gpt-5.2-lab``
# use a different endpoint without changing call sites.

model_dict = {
    'gpt-5.6': {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'base_url': 'https://api.openai.com/v1',
        'model_name': 'gpt-5.6',
    }
}


def _get_model_config(model_name):
    """Return provider settings for a configured model or OpenAI defaults."""
    config = model_dict.get(model_name)
    if config is None:
        # Preserve support for arbitrary OpenAI model names used by callers.
        config = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'base_url': 'https://api.openai.com/v1',
            'model_name': model_name,
        }

    if not config['api_key']:
        raise ValueError(f'No API key is configured for model {model_name!r}')
    if not config['base_url']:
        raise ValueError(
            f'No base URL is configured for model {model_name!r}; '
            'set DASHSCOPE_WORKSPACE_ID'
        )

    return config


def _request_headers(api_key):
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }


def load_prompts():
    """Load prompts from files.
    """
    prompts = dict()
    prompt_dir = os.path.join(CLASP_HOME, 'prompts')
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
                'content': meta_prompt
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_completion_tokens': 800
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

    model_config = _get_model_config(model_name)

    payload = prepare_inputs(message,
                             images,
                             meta_prompt=meta_prompt,
                             model_name=model_config['model_name'],
                             local_image=local_image)

    response = requests.post(
        f"{model_config['base_url'].rstrip('/')}/chat/completions",
        headers=_request_headers(model_config['api_key']),
        json=payload,
        timeout=120)
    response.raise_for_status()

    try:
        res = response.json()['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError('Model response has no valid message content') from error

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
                'content': meta_prompt
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_completion_tokens': 800
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

    model_config = _get_model_config(model_name)

    payload = prepare_inputs_incontext(
        message,
        images,
        meta_prompt=meta_prompt,
        model_name=model_config['model_name'],
        local_image=local_image,
        example_images=example_images,
        example_responses=example_responses)

    response = requests.post(
        f"{model_config['base_url'].rstrip('/')}/chat/completions",
        headers=_request_headers(model_config['api_key']),
        json=payload,
        timeout=120)
    response.raise_for_status()

    try:
        res = response.json()['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError('Model response has no valid message content') from error

    return res
