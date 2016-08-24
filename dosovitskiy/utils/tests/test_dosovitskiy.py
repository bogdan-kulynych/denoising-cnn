from dosovitskiy.utils.lowpassfilter import lowpassfilter
from dosovitskiy.utils.augment_patches_distr import sample_patches_multiscale
import dosovitskiy.stl10_input as stl
from dosovitskiy.imutils import imresize
import numpy as np

square_image = np.zeros((100, 100, 3))
square_image[25:51, 25:51, :] = 1
stl.DATA_PATH = "/home/mike/stl10_binary/unlabeled_X.bin"


SIGMA = 2
with open(stl.DATA_PATH) as f:
    im1 = stl.read_single_image(f)
    im2 = stl.read_single_image(f)
    im3 = stl.read_single_image(f)

IMRESIZE_PARAMETER_SETS = {
    'stl_1': [im1, 715737.0],
    'stl_2': [im2, 907880.0],
    'stl_3': [im3, 967787.0],
}

LOWPASSFILTER_PARAMETER_SETS = {
    'stl_1': [im1, 56232.89453125],
    'stl_2': [im2, 68211.6875],
    'stl_3': [im3, 65896.015625],
}

params = {
          'subsample_probmaps': 4,
          'sample_patches_per_image': 1,
          'patchsize':32,
          'num_patches':3,
          'one_patch_per_image': True,
          'scales': list(map(lambda x: pow(0.8,x),range(3,-1,-1))),
          'num_deformations': 150,
          'scale_range': [1/np.sqrt(2), np.sqrt(2)],
          'position_range': [-0.25, 0.25],
          'angle_range': [-20, 20],
          }

# Test to check if imresize works properly
def test_imresize_on_image():
    for image_name, image in IMRESIZE_PARAMETER_SETS.items():
        newim = imresize(image[0], scale=0.5)
        calcsum = np.sum(newim)
        assert abs(calcsum - image[1])/newim.size <= 0.1, '%s failed, %3.10f != %3.10f' % (image_name, calcsum, image[1])

# Test to check if a lowpassfilter behave same way as in Matlab
# with just summing resultant image
def test_lowpassfilter_on_image():
    for image_name, image in LOWPASSFILTER_PARAMETER_SETS.items():
        out = imresize(image[0][::4, ::4, :], 0.512).sum(axis=2)
        out = lowpassfilter(out, SIGMA)
        assert abs(out.sum() - image[1])/out.size <= 0.01, '%s failed, %3.10f != %3.10f' % (image_name, out.sum(), image[1])

# # Test to check if a sample_patches_multiscale works properly
# def test_sample_patches_multiscale():
#     np.random.seed(42)
#     images = np.stack((im1,im2,im3))
#     patches, pos = sample_patches_multiscale(images,params)
#     # assert abs(out.sum() - image[1])/out.size <= EPS, '%s failed, %3.10f != %3.10f' % (image_name, out.sum(), image[1])

if __name__ == '__main__':
    test_lowpassfilter_on_image()
