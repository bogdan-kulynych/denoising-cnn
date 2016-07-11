import math
import os
import pytest

import numpy as np

from keras.preprocessing.image import img_to_array, array_to_img

import utils.filters as filter_utils

from utils.preprocessing import FilterImageDataGenerator


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
        image_data = img_to_array(image)
        images = np.array([image_data] * N)
        data_generator.fit(images)

        X_batch, y_batch = next(data_generator.flow(images,
                batch_size=batch_size))
        assert X_batch.shape[0] == batch_size if N >= batch_size else N

    @pytest.mark.parametrize('filters', [
        ([filter_utils.Noop(),
          filter_utils.GaussianBlur()]),
        ([filter_utils.Noop(),
          filter_utils.GaussianBlur(),
          filter_utils.UniformNoise()]),
        ([filter_utils.Noop(),
          filter_utils.Lomo(),
          filter_utils.Vignette(),
          filter_utils.UniformNoise()])
    ])
    def test_flow_applies_filters(self, data_generator, image, filters):
        eps = 10e-5

        image_data = img_to_array(image)
        images = np.array([image_data])
        data_generator.fit(images)
        image_data = data_generator.standardize(image_data)

        flow = data_generator.flow(images, filters=filters,
                                   batch_size=len(filters),
                                   shuffle=True)
        X_batch, y_batch = next(flow)
        for i in range(flow.batch_size):
            X = X_batch[i]
            y = y_batch[i]
            if isinstance(filters[y], filter_utils.Noop):
                assert np.all(X - image_data) < eps
            else:
                assert np.any(X != image_data)

