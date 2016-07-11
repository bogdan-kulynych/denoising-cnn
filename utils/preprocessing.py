import os
import random
import numpy as np

from collections import defaultdict

from utils.filters import Noop

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import NumpyArrayIterator as BaseNumpyIterator
from keras.preprocessing.image import array_to_img, img_to_array


class FilterImageDataGenerator(ImageDataGenerator):
    '''Generate mini-batches applying filters in real-time'''

    def __init__(self, featurewise_center=True,
                 featurewise_std_normalization=True,
                 samplewise_center=False, samplewise_std_normalization=False,
                 dim_ordering=K.image_dim_ordering()):
        super().__init__(featurewise_center=featurewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
                samplewise_center=samplewise_center,
                samplewise_std_normalization=samplewise_std_normalization,
                dim_ordering=dim_ordering)

    def flow(self, X, filters=None, batch_size=32, shuffle=True,
             seed=None, save_to_dir=None, save_prefix='', save_format='jpeg'):
        if filters is None:
            filters = [Noop()]
        filters = list(filters)
        return FilterNumpyArrayIterator(
            X, y=None, image_data_generator=self, filters=filters,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory(self, directory,
                            color_mode='rgb', classes=None,
                            class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        raise NotImplemented('Flow from directory is not available for '
                             'FilterImageDataGenerator.')


class FilterNumpyArrayIterator(BaseNumpyIterator):
    def __init__(self, X, y, image_data_generator, filters,
                 batch_size=32, shuffle=True, seed=None,
                 dim_ordering=K.image_dim_ordering(),
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        self.filters = filters
        return super().__init__(
            X, y, image_data_generator,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format)

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # Instead of splitting indices 0, 1, ..., N into mini-batches,
        # we split "raw indices" 0, 1, ..., N * M, where M is the
        # number of filters to be applied to every image. Each raw
        # index r encodes an image index and an index of a filter
        # to apply, such that r = i*N + k, with i being the index of
        # an image, and k the index of a filter.
        nb_filters = len(self.filters)
        indices = super()._flow_index(N * nb_filters, batch_size, shuffle, seed)
        for raw_index_array, current_index, current_batch_size in indices:
            filter_index_array = raw_index_array // N
            index_array = raw_index_array % N
            yield (index_array, filter_index_array, current_index, current_batch_size)

    def next(self):
        # This is mostly copied over from Keras's NumpyArrayIterator:next
        with self.lock:
            index_array, filter_index_array, current_index, current_batch_size \
                    = next(self.index_generator)
        batch_X = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))

        for i, (j, k) in enumerate(zip(index_array, filter_index_array)):
            x = self.X[j]
            x = self.filters[k].apply(array_to_img(x))
            x = img_to_array(x)
            x = self.image_data_generator.standardize(x)
            batch_X[i] = x

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_X[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=current_index + i,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # Set y to filter ids
        batch_y = filter_index_array
        return batch_X, batch_y
