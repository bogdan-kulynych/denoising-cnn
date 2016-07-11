import os
import pytest
import numpy as np

from PIL import Image

from utils.preprocessing import FilterImageDataGenerator


DEFAULT_IMAGE_RPATH = 'sample_images/lena.jpg'
DEFAULT_IMAGE_DIR = 'sample_images'


@pytest.fixture(scope='module', params=[
    dict(loaded_from_file=False),
    dict(loaded_from_file=True),
])
def image(request):
    image_path = os.path.join(os.path.dirname(__file__), DEFAULT_IMAGE_RPATH)
    image = Image.open(image_path)
    image_data = np.asarray(image)
    if not request.param['loaded_from_file']:
        image = Image.fromarray(image_data)
    return Image.open(image_path)


@pytest.fixture(scope='module')
def image_directory(request):
    return os.path.join(os.path.dirname(__file__), DEFAULT_IMAGE_DIR)



