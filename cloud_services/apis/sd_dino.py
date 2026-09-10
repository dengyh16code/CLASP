import base64
import io

import numpy as np
import requests
from PIL import Image

from cloud_services.config import get_service_url


def convert_pil_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_array(encoded_array: str) -> np.ndarray:
    array_bytes = base64.b64decode(encoded_array)
    return np.load(io.BytesIO(array_bytes), allow_pickle=False)


class SDDINO:
    """Client for the Stable Diffusion and DINOv2 feature service."""

    def __init__(self, server_url=None, timeout: float = 120):
        self.server_url = (server_url or get_service_url("sd_dino")).rstrip("/")
        self.timeout = timeout

    def _process_image(self, endpoint: str, image: Image.Image) -> dict:
        response = requests.post(
            f"{self.server_url}/{endpoint}",
            json={"image": convert_pil_image_to_base64(image)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_features(self, image: Image.Image):
        """Return the fused SD+DINO feature map and the raw DINO feature map."""
        response = self._process_image("process_image", image)
        return (
            _decode_array(response["features"]),
            _decode_array(response["features_dino"]),
        )

    def get_dino_features(self, image: Image.Image) -> np.ndarray:
        """Return only the DINOv2 feature map."""
        response = self._process_image("process_image_dino", image)
        return _decode_array(response["features_dino"])
