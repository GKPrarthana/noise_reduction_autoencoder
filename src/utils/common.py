import os
from box.exceptions import BoxValueError
import yaml
from src import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import numpy as np
import cv2

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns its contents."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Creates list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves dictionary as JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON file into ConfigBox."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data as binary file."""
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads binary data."""
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Gets file size in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def decodeImage(imgstring: str, fileName: str):
    """Decodes base64 image string and saves to file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)

@ensure_annotations
def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """Encodes image file into base64 string."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

@ensure_annotations
def add_noise(image: np.ndarray, noise_type: str = "gaussian", noise_factor: float = 0.1) -> np.ndarray:
    """Adds noise to an image."""
    if noise_type == "gaussian":
        noisy = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    elif noise_type == "salt_pepper":
        noisy = image.copy()
        num_salt = int(np.ceil(noise_factor * image.size * 0.5))
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy[coords[0], coords[1]] = 1.0
        num_pepper = int(np.ceil(noise_factor * image.size * 0.5))
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy[coords[0], coords[1]] = 0.0
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    noisy = np.clip(noisy, 0.0, 1.0)
    logger.info(f"Added {noise_type} noise with factor {noise_factor}")
    return noisy

@ensure_annotations
def save_image_triplet(noisy: np.ndarray, clean: np.ndarray, denoised: np.ndarray, output_path: Path):
    """Saves noisy, clean, and denoised images side by side."""
    triplet = np.hstack((noisy.squeeze(), clean.squeeze(), denoised.squeeze()))
    cv2.imwrite(str(output_path), triplet * 255)
    logger.info(f"Saved image triplet at: {output_path}")