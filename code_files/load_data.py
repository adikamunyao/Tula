import os
import urllib.request
import tarfile
import tempfile

url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
download_folder = tempfile.gettempdir()
filename = os.path.join(download_folder, "flower_dataset.tgz")

image_folder = os.path.join(download_folder, "flower_photos")
if not os.path.exists(image_folder):
    print("Downloading Flowers data set (218 MB)...")
    urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(download_folder)
