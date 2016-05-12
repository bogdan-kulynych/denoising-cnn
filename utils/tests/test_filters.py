import os
import itertools

import numpy as np
import pytest

from PIL import Image

from utils.filters import Noop, GaussianBlur, GaussianNoise, UniformNoise


@pytest.fixture()
def image(scope='module'):
    image_path = os.path.join(os.path.dirname(__file__), 'lena.jpg')
    return Image.open(image_path)


GAUSSIAN_BLUR_PARAMS = [1, 2, 5]
GAUSSIAN_NOISE_PARAMS = list(itertools.product(
    [0.1, 0.5, 0.9],
    [0, 128, 255],
    [10, 100, 200]))
UNIFORM_NOISE_PARAMS = [0.1, 0.5, 0.9]


# Very basic test to check if a filter changes anything in
# an image
@pytest.mark.parametrize("filter_instance",
        [GaussianBlur(radius) for radius in GAUSSIAN_BLUR_PARAMS] +
        [GaussianNoise(*args) for args in GAUSSIAN_NOISE_PARAMS] +
        [UniformNoise(alpha) for alpha in UNIFORM_NOISE_PARAMS])
def test_filter_changes_image(filter_instance, image):
    image_after_filter = filter_instance.apply(image)
    assert image.tobytes() != image_after_filter.tobytes()

