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
    full_command = ' '.join([
        'convert',
        input_filename,
        command,
        output_filename])
    try:
        subprocess.run(full_command, shell=True, check=True)
    except:
        raise RuntimeError('Error when running `{}`'.format(
            full_command))
    result = Image.open(output_filename)
    return result


def _format_rgba(color, alpha):
    """
    >>> _format_rgba((128, 128, 128), 0.5)
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

    def __repr__(self):
        return 'Vignette({self.color}, {self.alpha})'.format(self=self)
    
    
class Gotham(object):
    def __init__(self, color='#222b6d', gamma=0.5):
        self.color = color
        self.gamma = gamma
    
    def apply(self, image):
        command = ("-modulate 120,10,100 "
                   "-fill '{color}' " 
                   " -colorize 20 " 
                   " -gamma {gamma} -contrast")
        
        return _imagemagick(image, command.format(color=self.color, gamma=self.gamma))
    
    def __perp__(self):
        return 'Gotham({self.color}, {self.gamma})'.format(self=self)
        
        
class Kelvin(object):
    def __init__(self):
        pass

    def apply(self, image):
        width, height = image.size
        color = _format_rgba((255, 153, 0), 0.5)
        command = ("-auto-gamma -modulate 120,50,100 "
                   "-size {width}x{height} -fill {color} "
                   "-draw 'rectangle 0,0 {width},{height}' -compose multiply ")
        return _imagemagick(image, command.format(
            color=color, width=width, height=height))

    def __repr__(self):
        return 'Kelvin({self.color}, {self.alpha})'.format(self=self)

        
class Lomo(object):
    def __init__(self, rlvl = 33, glvl = 33):
        self.rlvl = rlvl
        self.glvl = glvl
    
    def apply(self, image):
        command = ("-channel R -level {rlvl}% "
                   "-channel G -level {glvl}%")
        
        return _imagemagick(image, command.format(rlvl = self.rlvl, glvl = self.glvl))
                            
                            
    def __perp__(self):
        return 'Lomo({self.rlvl}, {self.glvl})'.format(self=self)
    
   
class Toaster(object):
    def __init__(self, brightness=150, saturation=80, hue=100, gamma=1.2):
        self.brightness = brightness
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma


    def apply(self, image):
        width, height = image.size
        command = ("-modulate {brightness},{saturation},{hue} "
                   "-gamma {gamma} "
                   "-contrast "
                   "-size {width}x{height} "
                   "radial-gradient:#cdc1c5-#ff9966 "
                   "-gravity center -compose multiply -flatten "
                   "-fill 'rgba(20,0,0)' -colorize 30% "
                  )
        
        return _imagemagick(image, command.format(
            brightness=self.brightness, saturation=self.saturation, hue=self.hue, gamma=self.gamma, filename = image.filename, width=width, height=height))

    def __repr__(self):
        return 'Toaster({self.brightness}, {self.saturation}, {self.hue}, {self.gamma})'.format(self=self)

  
        
        
        