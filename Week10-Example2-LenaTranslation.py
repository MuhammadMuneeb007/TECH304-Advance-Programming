import urllib.request
import urllib.error
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/lenna_RGB.tif",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
]


def download_lena(target_dir: Path) -> Path:
    """Download Lena image to target_dir and return the local path."""
    for url in DOWNLOAD_URLS:
        filename = url.rsplit("/", 1)[-1]
        target_path = target_dir / filename
        if target_path.exists():
            return target_path
        try:
            urllib.request.urlretrieve(url, target_path)
            return target_path
        except urllib.error.URLError:
            continue
    raise RuntimeError("Failed to download Lena image. Check internet or URLs.")


script_dir = Path(__file__).resolve().parent
image_path = download_lena(script_dir)

image = cv.imread(str(image_path))
if image is None:
    raise RuntimeError(f"Failed to load image from {image_path}")

h, w = image.shape[:2]
half_height, half_width = h // 4, w // 8
translation_matrix = np.float32([[1, 0, half_width], [0, 1, half_height]])
img_translation = cv.warpAffine(image, translation_matrix, (w, h))

plt.figure(figsize=(8, 6))
plt.imshow(cv.cvtColor(img_translation, cv.COLOR_BGR2RGB))
plt.title("Translation")
plt.axis("off")
plt.show()
