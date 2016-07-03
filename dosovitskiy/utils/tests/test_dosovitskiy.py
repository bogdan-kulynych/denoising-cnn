from dosovitskiy.utils.lowpassfilter import lowpassfilter
import numpy as np

square_image = np.zeros((100, 100, 3))
square_image[25:51, 25:51, :] = 1

SIGMA = 10
EPS = 1e-9
LOWPASSFILTER_PARAMETER_SETS = {
    'square_image': [square_image, 2024.3808457561],
    'zero_image': [np.zeros((100, 100, 3)), 0],
    'ones_image': [np.ones((100, 100, 3)), 100 * 100 * 3],
}


# Test to check if a lowpassfilter behave same way as in Matlab
# with just summing resultant image
def test_lowpassfilter_on_image():
    for image_name, image in LOWPASSFILTER_PARAMETER_SETS.items():
        calcsum = np.sum(lowpassfilter(image[0], SIGMA))
        assert abs(calcsum - image[1]) <= EPS, '%s failed, %3.10f != %3.10f' % (image_name, calcsum, image[1])


if __name__ == '__main__':
    test_lowpassfilter_on_image()
