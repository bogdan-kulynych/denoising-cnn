from dosovitskiy.stl10_input import read_all_images
from dosovitskiy.stl10_input import plot_image
from dosovitskiy.utils.lowpassfilter import lowpassfilter
from dosovitskiy.imutils import imresize

from math import pow
from tqdm import tqdm

from scipy.misc import imrotate
from scipy.io import loadmat

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import permutation, rand, randn

from skimage.color import hsv2rgb, rgb2hsv

import os


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

            borderwidth = (np.ceil(patchsize / subsample_probmaps / 2) + 1).astype(np.int)
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

    maskradius = np.floor(patchsize / subsample_probmaps).astype(np.int)
    mask = np.zeros((2 * maskradius + 1, 2 * maskradius + 1))

    npatch = 0
    pos = {}
    pbar = tqdm(total=num_patches, desc='Sampling patches: ')
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
            currpos['xc'] = ((currx * subsample_probmaps + 1) / currpos['scale_value'])[0] - 1
            currpos['yc'] = ((curry * subsample_probmaps + 1)/ currpos['scale_value'])[0] - 1
            currpos['patchsize'] = patchsize / currpos['scale_value']

            x1 = np.round(currpos['xc'] - np.floor(currpos['patchsize'] / 2)).astype(np.int)
            x2 = np.round(x1 + currpos['patchsize']).astype(np.int)
            y1 = np.round(currpos['yc'] - np.floor(currpos['patchsize'] / 2)).astype(np.int)
            y2 = np.round(y1 + currpos['patchsize']).astype(np.int)
            patches[npatch, :, :, :] = imresize(selected_images[ncurrimage, x1:x2, y1:y2, :].squeeze(), cropped_height=patchsize, cropped_width=patchsize)

            npatch += 1
            for nscale in range(max(0, currpos['scale'] - 3), min(num_scales, currpos['scale'] + 4)):
                im0 = sampling_probmap[ncurrimage][nscale]
                coeff = scales[nscale] / scales[ncurrscale]
                x1 = (np.round(currx * coeff) - maskradius).astype(np.int)
                x2 = (np.round(currx * coeff) + maskradius).astype(np.int)
                y1 = (np.round(curry * coeff) - maskradius).astype(np.int)
                y2 = (np.round(curry * coeff) + maskradius).astype(np.int)
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
                    image_probs[ncurrimage] = 0
                else:
                    sampling_probmap[ncurrimage][nscale][x11:x21, y11:y21] = \
                        mask[x11 - x1:mask.shape[0] - x2 + x21, y11 - y1:mask.shape[1] - y2 + y21] * \
                        sampling_probmap[ncurrimage][nscale][x11:x21, y11:y21]

            pbar.update(1)
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
        return np.digitize(x, np.insert(np.cumsum(P), 0, 0) / np.sum(P)).astype(np.int)


def augment_position_scale_color(pos_in, params):
    # Augments the training set (given by positions and scales in a spatial
    # pyramid) by including some neighboring positions and scales

    if not 'scale_range' in params:
        params['scale_range'] = [0.8, 1.2]

    if not 'position_range' in params:
        params['position_range'] = [-0.2, 0.2]

    if not 'angle_range' in params:
        params['angle_range'] = [-20, 20]

    if not 'num_deformations' in params:
        params['num_deformations'] = 5

    pos_out = {}

    for fn in params:
        if fn.find('_range') != -1 and type(params[fn]) == list and len(params[fn]) == 2:
            params[fn] = np.stack([params[fn]] * (len(pos_in['xc']) * params['num_deformations']))
        if fn.find('_range') != -1 and type(params[fn]) == np.ndarray and params[fn].shape == (len(pos_in['xc']), 2):
            params[fn] = np.tile(params[fn].transpose((2,0,1)), (params['num_deformations'], 1, 1)).reshape((-1,2))

    for fn in pos_in:
        if fn.find('_deform') == -1:
            pos_out[fn] = np.repeat(pos_in[fn], params['num_deformations'])
        else:
            pos_out[fn] = pos_in[fn]

    selected_deformations = ['scale', 'xc', 'yc', 'angle'] #'lightness', 'power']

    for deform in selected_deformations:
        if not deform + '_deform' in pos_in:
            if deform == 'xc' or deform == 'yc':
                pos_out[deform + '_deform'] = rand(*pos_out['xc'].shape) * \
                    (params['position_range'][:,1] - params['position_range'][:,0]) + params['position_range'][:,0]
            else:
                pos_out[deform + '_deform'] = rand(*pos_out['xc'].shape) * \
                    (params[deform + '_range'][:,1] - params[deform + '_range'][:,0]) + params[deform + '_range'][:,0]

    pos_out['patchsize'] /= pos_out['scale_deform']
    pos_out['xc'] += pos_out['xc_deform'] * pos_out['patchsize']
    pos_out['yc'] += pos_out['yc_deform'] * pos_out['patchsize']
    pos_out['angle'] = pos_out['angle_deform']

    return pos_out


def get_patches_rotations(pos_in, params, images):
    patchsize = params['patchsize']
    nchannels = params['nchannels']

    patches_aug = np.zeros((pos_in['xc'].size, patchsize, patchsize, nchannels),dtype=np.uint8)
    npatch = 0

    patches_npos = np.zeros(pos_in['xc'].size, dtype=np.int32)
    unique_nimg = np.unique(pos_in['nimg'])
    for n in tqdm(range(unique_nimg.size), desc='Extracting patches: '):
        nimg = unique_nimg[n]
        curr_selection = np.where(pos_in['nimg'] == nimg)[0]
        currimg = images[nimg]
        if currimg.shape[2] < 3:
            currimg = np.tile(currimg[:,:,None],(1,1,3))

        for npos in curr_selection:
            curr_angle = pos_in['angle'][npos]
            ps = pos_in['patchsize'][npos]
            ps_rot = ps * np.sqrt(2) * np.cos(np.mod(curr_angle,90)*np.pi/180 - np.pi/4)
            x1 = np.max((np.round(pos_in['xc'][npos] - ps_rot/2),0))
            x2 = np.min((np.round(pos_in['xc'][npos] + ps_rot/2) + 1,currimg.shape[0]))
            y1 = np.max((np.round(pos_in['yc'][npos] - ps_rot/2),0))
            y2 = np.min((np.round(pos_in['yc'][npos] + ps_rot/2) + 1,currimg.shape[1]))
            if y2-y1 >= ps_rot-1 and x2-x1 >= ps_rot-1 and ps_rot > 0:
                patch_tmp = currimg[x1:x2, y1:y2]
                if curr_angle != 0:
                    patch_tmp_rot = imrotate(patch_tmp, curr_angle, 'bilinear')
                else:
                    patch_tmp_rot = patch_tmp
                patch_to_save = patch_tmp_rot[np.max((np.floor(ps_rot/2 - ps/2),0)):np.min((np.ceil(ps_rot/2 + ps/2) + 1,patch_tmp_rot.shape[0])),
                    np.max((np.floor(ps_rot/2 - ps/2),0)) : np.min((np.ceil(ps_rot/2 + ps/2) + 1,patch_tmp_rot.shape[1])), :]
                patches_aug[npatch] = imresize(patch_to_save,cropped_width=patchsize, cropped_height=patchsize).astype(np.uint8)
                patches_npos[npatch] = npos
                npatch += 1

    patches_aug = patches_aug[:npatch]
    patches_npos = patches_npos[:npatch]
    pos_out = {}
    for key in pos_in:
        pos_out[key] = pos_in[key][patches_npos]

    return patches_aug, pos_out


def color_transform(img_in, M):
    return np.sum(img_in[:,:,:,:,None].transpose((0,1,2,4,3)) * M[:,:,None,None,None].transpose((2,3,0,1,4)), axis = 2).squeeze()


def adjust_color(patches, pos, batchsize=10000):

    num_batches = np.ceil(patches.shape[3] / batchsize)

    min_in = np.min(patches).astype(np.float32)
    max_in = np.max(patches).astype(np.float32)
    mean_in = np.mean(patches, (1,2,0)).astype(np.float32)

    patches_out = np.zeros(patches.shape, np.uint8)

    matfile = loadmat('./color_eigenvectors.mat')
    v = matfile['v'].astype(np.float32)
    d = matfile['d'].astype(np.float32)

    color_deform = np.stack((pos['color1_deform'], pos['color2_deform'], pos['color3_deform']))

    for batch in np.arange(num_batches):
        n1 = batch * batchsize
        n2 = np.min(((batch + 1) * batchsize, patches.shape[0]))
        patches_batch = patches[n1:n2].astype(np.float32)
        patches_batch = patches_batch - mean_in
        patches_batch = (patches_batch - min_in) / (max_in - min_in)

        curr_min = np.min(patches_batch)
        curr_max = np.max(patches_batch)
        if patches.shape[3] == 1:
            patches_batch = np.tile(patches_batch, (1,1,1,3))
        pp1 = color_transform(patches_batch.transpose((1,2,3,0)), v)
        pp2 = pp1 * color_deform[:,n1:n2].reshape((1,1,3,pp1.shape[3]))
        pp3 = color_transform(pp2, np.linalg.inv(v))
        if patches.shape[2] == 1:
            pp3 = pp3.mean(axis=2)
        pp3[pp3 > curr_max] = curr_max
        pp3[pp3 < curr_min] = curr_min

        pp3 += ((mean_in - min_in) / (max_in - min_in))[None,None,:,None]

        pp3[pp3 < 0] = 0
        pp3[pp3 > 1] = 1

        patches_out[n1:n2, :] = (pp3.transpose((3,0,1,2)) * 255).astype(np.uint8)

    return patches_out


def hsv2rgb_batch(hsv_images):
    input_size = hsv_images.shape
    rgb_images = np.zeros(input_size, dtype=np.uint8)

    for n in range(input_size[0]):
        currimg = hsv_images[n]
        rgb_images[n] = (255*hsv2rgb(currimg)).astype(np.uint8)
    return rgb_images


def rgb2hsv_batch(rgb_images):
    input_size = rgb_images.shape
    hsv_images = np.zeros(input_size, dtype=np.float32)
    min_in = np.min(rgb_images)
    max_in = np.max(rgb_images)

    for n in range(input_size[0]):
        currimg = rgb_images[n]
        if currimg.dtype==np.float32:
            currimg = currimg.astype(np.float32)
        currimg = (currimg-min_in) / (max_in-min_in)
        hsv_images[n] = rgb2hsv(currimg)
    return hsv_images


def power_transform(in_images, params, batchsize=1000):
    num_batches = np.ceil(in_images.shape[3] / batchsize)
    out_images = np.zeros(in_images.shape, dtype=np.uint8)

    for batch in np.arange(num_batches):
        n1 = batch * batchsize
        n2 = np.min(((batch + 1) * batchsize, in_images.shape[0]))
        in_batch = in_images[n1:n2].astype(np.float32)
        in_batch -= np.min(in_batch, axis=(1, 2, 3))[:,None,None,None]
        max_in_batch = np.max(in_batch, axis=(1,2,3))
        in_batch /= np.maximum(max_in_batch, 1e-3)[:,None,None,None]
        hsv_batch = rgb2hsv_batch(in_batch)

        hsv_batch[:,:,:,2] **= params['v_power_deform'][n1:n2].reshape((n2 - n1,1,1))
        hsv_batch[:,:,:,2] *= params['v_mult_deform'][n1:n2].reshape((n2 - n1,1,1))
        hsv_batch[:,:,:,2] += params['v_add_deform'][n1:n2].reshape((n2 - n1,1,1))

        hsv_batch[:,:,:,1] **= params['s_power_deform'][n1:n2].reshape((n2 - n1,1,1))
        hsv_batch[:,:,:,1] *= params['s_mult_deform'][n1:n2].reshape((n2 - n1,1,1))
        hsv_batch[:,:,:,1] += params['s_add_deform'][n1:n2].reshape((n2 - n1,1,1))

        hsv_batch[:,:,:,0] = np.mod(hsv_batch[:,:,:,0] + params['h_add_deform'][n1:n2].reshape((n2 - n1,1,1)), 1)

        hsv_batch[hsv_batch < 0] = 0
        hsv_batch[hsv_batch > 1] = 1

        out_images[n1:n2] = hsv2rgb_batch(hsv_batch)

    return out_images

save_path = '/home/mike/stl_dosov/unlabeled_data_STL_16000.npz'

np.random.seed(42)

images = read_all_images('/home/mike/stl10_binary/unlabeled_X.bin')

params = {
    'subsample_probmaps': 4,
    'sample_patches_per_image': 1,
    'patchsize': 32,
    'num_patches': 16000,
    'one_patch_per_image': True,
    'scales': list(map(lambda x: pow(0.8, x), range(3, -1, -1))),
    'num_deformations': 150,
    'scale_range': [1 / np.sqrt(2), np.sqrt(2)],
    'position_range': [-0.25, 0.25],
    'angle_range': [-20, 20],
    'nchannels': 3,
}

params['num_patches'] = min(params['num_patches'], images.shape[0])

patches, pos = sample_patches_multiscale(images, params)

pos['cluster'] = np.arange(len(pos['xc']))
pos['detector'] = np.arange(len(pos['xc']))

pos_aug3 = pos

all_coeffs = [1, 1, 0.5, 2, 2, 0.5, 0.5, 0.1, 0.1, 0.1]

pos_aug4 = augment_position_scale_color(pos_aug3,params)

curr_selection = np.arange(pos_aug4['xc'].size)

xc_shape = pos_aug4['xc'].shape
pos_aug4['color1_deform'] = 2**(randn(*xc_shape)*all_coeffs[0])
pos_aug4['color2_deform'] = 2**(randn(*xc_shape)*all_coeffs[1])
pos_aug4['color3_deform'] = 2**(randn(*xc_shape)*all_coeffs[2])
pos_aug4['v_power_deform'] = 2**(all_coeffs[3]*(rand(*xc_shape)*2-1))
pos_aug4['s_power_deform'] = 2**(all_coeffs[4]*(rand(*xc_shape)*2-1))
pos_aug4['v_mult_deform'] = 2**(all_coeffs[5]*(rand(*xc_shape)*2-1))
pos_aug4['s_mult_deform'] = 2**(all_coeffs[6]*(rand(*xc_shape)*2-1))
pos_aug4['v_add_deform'] = all_coeffs[7]*(rand(*xc_shape)*2-1)
pos_aug4['s_add_deform'] = all_coeffs[8]*(rand(*xc_shape)*2-1)
pos_aug4['h_add_deform'] = all_coeffs[9]*(rand(*xc_shape)*2-1)

patches_aug5, pos_aug5 = get_patches_rotations(pos_aug4, params, images)

images = []

print('Augmenting color...')
patches_aug5 = adjust_color(patches_aug5, pos_aug5, 10000)
patches_aug5 = power_transform(patches_aug5, pos_aug5, 10000)

images = patches_aug5
rename = np.arange(np.max(pos_aug5['detector'])+1)
rename[np.unique(pos_aug5['detector'])] = np.arange(len(np.unique(pos_aug5['detector'])))
labels = rename[pos_aug5['detector']]
print('Saving to %s...', 'Not defined')
np.savez(save_path, images=images,labels=labels)

