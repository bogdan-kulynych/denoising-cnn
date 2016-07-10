import os
import pytest

from PIL import Image

from utils.preprocessing import FilterImageDataGenerator


DEFAULT_IMAGE_RPATH = 'sample_images/lena.jpg'
DEFAULT_IMAGE_DIR = 'sample_images'


@pytest.fixture(scope='module')
def image():
    image_path = os.path.join(os.path.dirname(__file__), DEFAULT_IMAGE_RPATH)
    return Image.open(image_path)


@pytest.fixture(scope='module')
def image_directory(request):
    return os.path.join(os.path.dirname(__file__), DEFAULT_IMAGE_DIR)



