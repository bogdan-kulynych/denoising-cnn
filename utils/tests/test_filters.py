import os
import numpy as np
import pytest

from PIL import Image

from utils.filters import Noop, GaussianBlur


@pytest.fixture()
def image(scope='module'):
    image_path = os.path.join(os.path.dirname(__file__), 'lena.jpg')
    return Image.open(image_path)


@pytest.mark.parametrize("filter_instance",
        # [Noop(), GaussianBlur(0)] +
        [GaussianBlur(radius) for radius in [1, 2, 5, 10]]
    )
def test_filter_changes_image(filter_instance, image):
    altered_image = filter_instance.apply(image)
    assert image.tobytes() != altered_image.tobytes()

