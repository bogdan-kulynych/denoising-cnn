import math
import os
import pytest

import numpy as np

from keras.preprocessing.image import img_to_array

import utils.filters as filters

from utils.preprocessing import FilterImageDataGenerator


def calculate_nb_files(directory):
    result = 0
    for root, dirs, files in os.walk(directory):
        result += len(files)
    return result


@pytest.mark.usefixtures('image_directory')
class TestFilterImageDataGenerator:
    @pytest.fixture(params=[
        dict(featurewise_center=True, featurewise_std_normalization=True),
        dict(featurewise_center=False, featurewise_std_normalization=False),
    ], scope='function')
    def data_generator(self, request):
        kwargs = request.param
        return FilterImageDataGenerator(**kwargs)

    @pytest.mark.parametrize("N,batch_size", [
        (1, 10), (10, 10), (10, 5), (10, 3), (10, 2)
    ])
    def test_flow_batch_size(self, data_generator, image,
                             N, batch_size):
        images = np.array([img_to_array(image)] * N)
        data_generator.fit(images)
        batch = next(data_generator.flow(images, batch_size=batch_size))
        assert batch.shape[0] == batch_size if N >= batch_size else N

    @pytest.mark.parametrize('filters,expected_changed,expected_not_changed', [
        ([filters.Noop(), filters.GaussianBlur()], 1, 1),
        ([filters.Noop(), filters.GaussianBlur(), filters.UniformNoise()], 2, 1)
    ])
    def test_flow_applies_filters(self, data_generator, image, filters,
                                  expected_changed, expected_not_changed):
        eps = 10e-5
        np.random.seed(1)

        image_data = img_to_array(image)
        images = np.array([image_data])
        data_generator.fit(images)

        flow = data_generator.flow(images, filters=filters,
                                   batch_size=len(filters),
                                   shuffle=True)
        batch = next(flow)
        not_changed_images = 0
        changed_images = 0

        image_data = data_generator.standardize(image_data)
        for i in range(flow.batch_size):
            generated_image = batch[i, :]
            if np.all(np.abs(generated_image - image_data) < eps):
                not_changed_images += 1
            else:
                changed_images += 1

        assert not_changed_images == expected_not_changed
        assert changed_images == expected_changed

