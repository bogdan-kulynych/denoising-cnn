import io
import subprocess
import tempfile

import numpy as np

from scipy import stats
from PIL import Image, ImageFilter


def _imagemagick(image, command):
    """
    Execute `convert <image path> <command> <output path>` via subprocess,
    load and return the resulting image
    """
    try:
        input_filename = image.filename
    except:
        raise TypeError('Image needs to be loaded from file')
    with tempfile.NamedTemporaryFile() as fp:
        output_filename = fp.name
    expanded_command = ' '.join([
        'convert',
        input_filename,
        command,
        output_filename])
    try:
        subprocess.run(expanded_command, shell=True, check=True)
    except:
        raise RuntimeError('Error when running `{}`'.format(
            expanded_command))
    result = Image.open(output_filename)
    return result


def _format_rgba(color, alpha):
    """
    >>> _rgba((128, 128, 128), 0.5)
    "'rgba(128,128,128,0.5)'"
    """
    r, g, b = color
    return "'rgba({r:3d},{g:3d},{b:3d},{a:0.1f})'".format(r=r, g=g, b=b, a=alpha)


class Noop(object):
    def apply(self, image):
        return image.copy()


class GaussianBlur(object):
    def __init__(self, radius):
        self.radius = radius

    def apply(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.radius))

    def __repr__(self):
        return 'GaussianBlur({self.radius})'.format(self=self)


class GaussianNoise(object):
    def __init__(self, alpha=0.5, loc=0, scale=255):
        self.alpha = alpha
        self.loc = loc
        self.scale = scale
        self.dist = stats.truncnorm(0, 255, loc, scale)

    def apply(self, image):
        width, height = image.size
        noise = np.uint8(self.dist.rvs(size=(width, height, 3)))
        overlay = Image.fromarray(noise)
        result = Image.blend(image, overlay, self.alpha)
        return result

    def __repr__(self):
        return ('GaussianNoise('
               '{self.alpha}, {self.loc}, {self.scale})').format(self=self)


class UniformNoise(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def apply(self, image):
        width, height = image.size
        noise = np.uint8(np.random.random(size=(width, height, 3)) * 255)
        overlay = Image.fromarray(noise)
        result = Image.blend(image, overlay, self.alpha)
        return result

    def __repr__(self):
        return 'UniformNoise({self.alpha})'.format(self=self)


class Vignette(object):
    def __init__(self, color=(0, 0, 0), alpha=0.9):
        self.color = color
        self.alpha = alpha

    def apply(self, image):
        width, height = image.size
        color = _format_rgba(self.color, self.alpha)
        command = ('-size {width}x{height} '
                   'radial-gradient:none-{color} '
                   '-gravity center -compose multiply -flatten ')
        return _imagemagick(image, command.format(
            color=color, width=width, height=height))
