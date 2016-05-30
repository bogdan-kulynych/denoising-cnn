import os
import pytest

from PIL import Image


@pytest.fixture()
def image(scope='module'):
    image_path = os.path.join(os.path.dirname(__file__), 'lena.jpg')
    return Image.open(image_path)

