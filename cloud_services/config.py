from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def get_service_url(service_name: str) -> str:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        service = config["services"][service_name]
        scheme = service.get("scheme", "http")
        host = service["host"]
        port = service["port"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Missing or invalid configuration for service '{service_name}'"
        ) from error

    return f"{scheme}://{host}:{port}"
