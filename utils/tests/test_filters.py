import itertools
import pytest

from utils.filters import Noop, GaussianBlur, GaussianNoise, UniformNoise, \
                          Vignette


FILTER_PARAMETER_SETS = {
    # Noop: [()],
    GaussianBlur: [1, 2, 5],
    GaussianNoise: itertools.product(
        (0.1, 0.5, 0.9),
        (0, 128, 255),
        (10, 100, 200)),
    UniformNoise: [0.1, 0.5, 0.9],
    Vignette: [
        ((0, 0, 0), 0.5),
        ((255, 255, 255), 0.75)],
}


def generate_filter_instances(filter_param_sets):
    """
    Generate a list of filters instantiated with corresponding
    params from filter_param_sets
    """
    filter_instances = []
    for filter_class, param_set in filter_param_sets.items():
        batch = []
        for params in param_set:
            try:
                len(params)
            except TypeError:
                params = [params]
            batch.append(filter_class(*params))
        filter_instances.extend(batch)
    return filter_instances


# Very basic test to check if a filter changes anything in
# an image
@pytest.mark.parametrize("filter_instance",
        generate_filter_instances(FILTER_PARAMETER_SETS))
def test_filter_changes_image(filter_instance, image):
    image_after_filter = filter_instance.apply(image)
    assert image.tobytes() != image_after_filter.tobytes()

