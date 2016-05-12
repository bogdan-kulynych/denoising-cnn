import numpy as np

from scipy import stats
from PIL import Image, ImageFilter


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
