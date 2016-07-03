import numpy as np
from numpy.fft import fft2, fftshift, ifft2
from skimage.filters import gabor_kernel
from math import cos, sin


def lowpassfilter(img, sigma, batchsize=100, type='gaussian', verbose='false'):
    imsize = img.shape
    if img.shape[0] > 1 and img.shape[1] > 1:
        img = np.reshape(img, (imsize[0], imsize[1], -1))
        out = np.zeros(img.shape, dtype=img.dtype)

        dx = round(2 * sigma)
        tmpimg = np.concatenate((img[:, dx:0:-1, :], img, img[:, -1:-dx-1:-1, :]), axis=1)
        tmpimg = np.concatenate((tmpimg[dx-1::-1, :, :], tmpimg, tmpimg[img.shape[0]:img.shape[0]-dx-1:-1, :, :]))
        if verbose:
            print('\nPreparing the filter...')
        if type == 'gaussian':
            lowpass = gabor_atom([sigma, sigma], 0, 0, 0, 0, tmpimg.shape[0], tmpimg.shape[1], 1).astype(img.dtype)
            lowpass = fft2(fftshift(lowpass))
        else:
            raise NotImplementedError('lowpass for flattened image')

        nbatches = np.ceil(img.shape[2] / batchsize)

        m = 0
        for batch in np.arange(nbatches):
            if verbose:
                print('\n  Filtering batch %d/%d...' % (batch, nbatches))
            n = m
            m = min(m + batchsize, img.shape[2]) # !!! something bad could happen (identical to matlab, indexing problems expected)
            tmplowpass = np.real(ifft2(fft2(tmpimg[:, :, n:m],axes=(1,0))*lowpass[:, :, None],axes=(1,0)))
            out[:, :, n:m] = tmplowpass[dx:dx + img.shape[0], dx:dx + img.shape[1], :]

        out = np.reshape(out, imsize)
        print(np.sum(out[:, :, n:m]))
    else:
        raise NotImplementedError('lowpass for flattened image')

    return out

def gabor_atom(sigma, phi, frequency, position_x, position_y, windowsize_x, windowsize_y, normalize=0):
    # create coordinate grid
    halfsize_x = (windowsize_x - 1) / 2
    halfsize_y = (windowsize_y - 1) / 2
    x, y = np.meshgrid(np.arange(-halfsize_y - position_y, halfsize_y - position_y + 1),
                       np.arange(-halfsize_x - position_x, halfsize_x - position_x + 1))

    # rotated coordinate grid
    x1 = x * cos(phi) + y * sin(phi)
    y1 = -x * sin(phi) + y * cos(phi)

    # the Gaussian filter
    gauss = np.exp(-x1 ** 2 / sigma[1] ** 2 / 2 - y1 ** 2 / sigma[0] ** 2 / 2)

    # multiply by complex exponent and normalize to have zero mean (if frequency>0)
    if abs(frequency) > 1e-4:
        int1 = np.mean(gauss)
        out = gauss * np.exp(1j * x1 * frequency)
        int2 = np.mean(out)
        out = out - int2 / int1 * gauss
    else:
        out = gauss * np.exp(1j * x1 * frequency)

    # normalize to have L1 norm 1
    if normalize:
        out = out / np.sum(abs(out))

    return out
