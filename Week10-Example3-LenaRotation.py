import urllib.request
import urllib.error
from pathlib import Path

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

image = Image.open(image_path).convert("RGB")

angle = 180
scale = 0.1

rotated = image.rotate(angle, expand=True)
scaled_size = (
    max(1, int(rotated.width * scale)),
    max(1, int(rotated.height * scale)),
)
rotated_scaled = rotated.resize(scaled_size)

plt.figure(figsize=(10, 8))
plt.imshow(rotated_scaled)
plt.title("Rotation")
plt.axis("off")
plt.show()
