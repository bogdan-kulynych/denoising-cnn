from dosovitskiy.stl10_input import read_all_images
from dosovitskiy.utils.sample_patches_multiscale import sample_patches_multiscale
from math import pow
from numpy.random import rand, randn
import numpy as np


class Params:
    pass

images = read_all_images('/home/mike/stl10_binary/train_X.bin')[:10,:]

params = {
          'subsample_probmaps': 4,
          'sample_patches_per_image': 1,
          'patchsize':32,
          'num_patches':16000,
          'one_patch_per_image': True,
          'scales': list(map(lambda x: pow(0.8,x),range(3,-1,-1))),
          'num_deformations': 150,
          'scale_range': [1/np.sqrt(2), np.sqrt(2)],
          'position_range': [-0.25, 0.25],
          'angle_range': [-20, 20],

          }

params['num_patches'] = min(params['num_patches'], images.shape[0])


patches, pos = sample_patches_multiscale(images, params)

pos.cluster = np.arange(len(pos['xc']))
pos.detector = np.arange(len(pos['xc']))

pos_aug3 = pos

all_coeffs = [1, 1, 0.5, 2, 2, 0.5, 0.5, 0.1, 0.1, 0.1]

pos_aug4 = augment_position_scale_color(pos_aug3,params)

curr_selection =  np.arange(len(pos_aug4['xc']))

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

patches_aug5, pos_aug5 = get_patches_rotations(pos_aug4, params, image_names)

print('Augmenting color...')
patches_aug5 = adjust_color(patches_aug5, pos_aug5, 10000)
patches_aug5 = power_transform(patches_aug5, pos_aug5, 10000)

images = patches_aug5
rename = np.arange(max(pos_aug5.detector))
rename[np.unique(pos_aug5['detector'])] = np.arange(len(np.unique(pos_aug5['detector']))-1)
labels = np.reshape(rename[pos_aug5['detector']],(-1,1))
print('Saving to %s...', 'Not defined')
# save(save_path,'images','labels', '-v7.3')