import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from PIL import Image
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

# Load image and convert to RGB array
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Simple grayscale conversion
gray = np.mean(image_np, axis=2).astype(np.uint8)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.tight_layout()
plt.show()

# Split and display RGB channels
red = image_np[:, :, 0]
green = image_np[:, :, 1]
blue = image_np[:, :, 2]

plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
plt.imshow(red, cmap="Reds")
plt.title("Red")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(green, cmap="Greens")
plt.title("Green")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blue, cmap="Blues")
plt.title("Blue")
plt.axis("off")

plt.tight_layout()
plt.show()

