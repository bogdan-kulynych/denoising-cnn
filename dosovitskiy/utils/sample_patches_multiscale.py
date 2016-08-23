import numpy as np

from tqdm import tqdm
from numpy.random import permutation
from dosovitskiy.imutils import imresize
import matplotlib.pyplot as plt

from dosovitskiy.utils.lowpassfilter import lowpassfilter


def sample_patches_multiscale(images, params, selected_images=[]):
    scales = params['scales']
    subsample_probmaps = params['subsample_probmaps']
    patchsize = params['patchsize']
    sampling_probmap = []
    num_scales = len(scales)
    assert (patchsize % subsample_probmaps != 0, 'subsampling_probmaps must divide patchsize')

    # reading in a subset of images
    if len(selected_images) == 0:
        if params['one_patch_per_image']:
            num_selected_images = params['num_patches']
        else:
            num_selected_images = min(16000, params['num_patches'])

        orig_image_num = permutation(images.shape[0])[:num_selected_images]
        selected_images = images[orig_image_num, :]
    else:
        orig_image_num = range(len(selected_images))

    num_selected_images = len(selected_images)

    image_probs = np.zeros(num_selected_images)
    for i in tqdm(range(num_selected_images), desc='Calculating probability maps: '):
        scales_list = []
        for nscale in scales:
            # returns image smaller than in Matlab
            im = imresize(selected_images[i, ::subsample_probmaps, ::subsample_probmaps, :], nscale)

            # plt.show()
            # plt.imshow(im)

            im = np.sum(im, axis=2)
            im = im - lowpassfilter(im, 2)

            # plt.imshow(im)

            energy_radius = min(patchsize / subsample_probmaps / 4, im.shape[0] / 4)
            im = lowpassfilter(im ** 2, energy_radius)

            # plt.imshow(im)

            borderwidth = np.ceil(patchsize / subsample_probmaps / 2) + 1
            im[:borderwidth, :] = 0
            im[-borderwidth:, :] = 0
            im[:, :borderwidth] = 0
            im[:, -borderwidth:] = 0
            im[im < 0] = 0

            # plt.imshow(im)

            image_probs[i] = np.sum(im) / (im.shape[0] - 2 * borderwidth) / (im.shape[1] - 2 * borderwidth)
            scales_list.append(im)
        sampling_probmap.append(scales_list)
    scale_probs = np.ones(num_scales)

    # Sampling patches according to these probability maps
    num_patches = params['num_patches']
    patches = np.zeros((num_patches, patchsize, patchsize, 3), dtype='uint8')

    maskradius = np.floor(patchsize / subsample_probmaps)
    mask = np.zeros((2 * maskradius + 1, 2 * maskradius + 1))

    npatch = 0
    pos = {}
    pbar = tqdm(range(num_patches), desc='Sampling patches: ')
    while npatch < num_patches:
        ncurrscale = randp(scale_probs, 1) - 1
        ncurrimage = randp(image_probs, 1) - 1
        im = sampling_probmap[ncurrimage][ncurrscale]
        if np.any(im > 0):
            currpos = {}
            currinds = randp(im.ravel(), 1) - 1
            currx, curry = np.unravel_index(currinds,im.shape, order='F')

            currpos['nimg'] = orig_image_num[ncurrimage][0]
            currpos['scale'] = ncurrscale[0]
            currpos['scale_value'] = scales[currpos['scale']]
            currpos['xc'] = (currx * subsample_probmaps / currpos['scale_value'])[0]
            currpos['yc'] = (curry * subsample_probmaps / currpos['scale_value'])[0]
            currpos['patchsize'] = patchsize / currpos['scale_value']

            x1 = np.round(currpos['xc'] - np.floor(currpos['patchsize'] / 2))
            x2 = np.round(x1 + currpos['patchsize'] )
            y1 = np.round(currpos['yc'] - np.floor(currpos['patchsize'] / 2))
            y2 = np.round(y1 + currpos['patchsize'])
            patches[npatch, :, :, :] = imresize(selected_images[ncurrimage, x1:x2, y1:y2, :].squeeze(), cropped_height=patchsize, cropped_width=patchsize)

            npatch += 1
            for nscale in range(max(0, currpos['scale'] - 3), min(num_scales, currpos['scale'] + 3)):
                im0 = sampling_probmap[ncurrimage][nscale]
                coeff = scales[nscale] / scales[ncurrscale]
                x1 = np.round(currx * coeff) - maskradius
                x2 = np.round(currx * coeff) + maskradius
                y1 = np.round(curry * coeff) - maskradius
                y2 = np.round(curry * coeff) + maskradius
                x11 = np.max((x1, 0))
                x21 = np.min((x2, im0.shape[0]))
                y11 = np.max((y1, 0))
                y21 = np.min((y2, im0.shape[1]))

                if im0[x11:x21, y11:y21].shape != mask[x11 - x1:mask.shape[0] - x2 + x21 - 1,
                                                       y11 - y1:mask.shape[1] - y2 + y21 - 1].shape:
                    print(im0[x11:x21, y11:y21].shape)
                    print(mask[x11 - x1:mask.shape[0] - x2 + x21, y11 - y1:mask.shape[1] - y2 + y21])
                    print([x1, x2, y1, y2, x11, x21, y11, y21])

                if params['one_patch_per_image']:
                    sampling_probmap[ncurrimage][nscale] = 0  # Don't sample from the same image twice!
                else:
                    sampling_probmap[ncurrimage][nscale][x11:x21, y11:y21] = \
                        mask[x11 - x1:mask.shape[0] - x2 + x21, y11 - y1:mask.shape[1] - y2 + y21] * \
                        sampling_probmap[ncurrimage][nscale][x11:x21, y11:y21]

            pbar.update(npatch)
            for key in currpos:
                if key in pos:
                    pos[key] = pos[key] + [currpos[key]]
                else:
                    pos[key] = [currpos[key]]
    return patches, pos


def show_cell(image_list, finsize=[64, 48]):
    imtoshow = np.zeros(tuple(finsize) + (3, len(image_list)))
    for i in range(len(image_list)):
        imtoshow[:, :, :, i] = imresize(image_list[i], finsize)
    plt.imshow(imtoshow)


def randp(P, shape):
    x = np.random.rand(shape)
    if any(P < 0):
        raise('All probabilities should be 0 or larger.')

    if P.size == 0 or np.sum(P) == 0:
        Warning(':ZeroProbabilities', 'All zero probabilities')
        return np.zeros(x.shape)
    else:
        return np.digitize(x, np.insert(np.cumsum(P), 0, 0) / np.sum(P))

