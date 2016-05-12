from PIL import ImageFilter


class GaussianBlur(object):
    def __init__(self, radius):
        self.radius = radius

    def apply(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.radius))

