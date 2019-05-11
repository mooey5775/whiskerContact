import numpy as np

from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils

from keras.backend import permute_dimensions

import keras.backend as K

def _find_maxima(x):

    x = K.cast(x, K.floatx())

    col_max = K.max(x, axis=1)
    row_max = K.max(x, axis=2)

    maxima = K.max(col_max, 1)
    maxima = K.expand_dims(maxima, -2)

    cols = K.cast(K.argmax(col_max, -2), K.floatx())
    rows = K.cast(K.argmax(row_max, -2), K.floatx())
    cols = K.expand_dims(cols, -2)
    rows = K.expand_dims(rows, -2)

    # maxima = K.concatenate([rows, cols, maxima], -2) # y, x, val
    maxima = K.concatenate([cols, rows, maxima], -2) # x, y, val

    return maxima

def find_maxima(x, data_format):
    """Finds the 2D maxima contained in a 4D tensor.
    # Arguments
        x: Tensor or variable.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
        x = _find_maxima(x)
        x = permute_dimensions(x, [0, 2, 1])
        return x
    elif data_format == 'channels_last':
        x = _find_maxima(x)
        return x
    else:
        raise ValueError('Invalid data_format:', data_format)

class Maxima2D(Layer):
    """Maxima layer for 2D inputs.
    Finds the maxima and 2D indices
    for the channels in the input.
    The output is ordered as [row, col, maximum].
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        3D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, 3, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, 3)`
    """

    def __init__(self, data_format=None, **kwargs):
        super(Maxima2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    3)
        elif self.data_format == 'channels_last':
            return (input_shape[0],
                    3,
                    input_shape[3])

    def call(self, inputs):
        return find_maxima(inputs, self.data_format)

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Maxima2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _cropFrame(image, obsPos, size, cropSize, mean, std):
    cropDia = K.cast(cropSize/2, "int32")
    x = K.min([K.max([obsPos[0], 1]), size[1]-2])
    y = K.min([K.max([obsPos[1], 1]), size[0]-2])
    newImg = image[K.max([0,y-cropDia]):K.min([y+cropDia, size[0]-1]), K.max([0,x-cropDia]):K.min([x+cropDia, size[1]-1])]
    newImg = K.concatenate([(K.zeros((K.shape(newImg)[0],K.max([0, cropDia-x]),size[2]))-mean)/std,newImg],axis=1)
    newImg = K.concatenate([(K.zeros((K.max([0, cropDia-y]), K.shape(newImg)[1], size[2]))-mean)/std, newImg], axis=0)
    newImg = K.concatenate([newImg, (K.zeros((K.max([0, cropDia+1-size[0]+y]),K.shape(newImg)[1],size[2]))-mean)/std], axis=0)
    newImg = K.concatenate([newImg, (K.zeros((K.shape(newImg)[0],K.max([0, cropDia+1-size[1]+x]),size[2]))-mean)/std], axis=1)
    return newImg

def _getCrop(x, size, cropSize, mean, std, wOffset, hOffset, i):
    image = x[1][i]
    obsPos = [K.cast(x[0][i][0][0], "int32")+wOffset, K.cast(x[0][i][1][0], "int32")+hOffset]
    return _cropFrame(image, obsPos, size, cropSize, mean, std)

class PointCrop2D(Layer):
    """ Layer that crops a window centered at a point with some crop_size. If the crop goes outside the frame, it's filled with 0s.
    
    It takes as input a batch of points with shape (batch, channels, 3), where the third dimension contains [W, H, maxima] and a 4D set of images assumed to be NHWC (ha go change the code yourself).
    
    Uses maxima from first channel to crop the entire frame.
    
    ONLY USE EVEN CROP SIZES
    
    Input shape validation is very sketchy right now (mostly not there)
    
    Arguments:
        cropSize: height and width of crop (always square)
        mean (0): image mean value (to normalize filled 0s)
        std (1): image standard deviation (to normalize filled 0s)
        hOffset (0): how much to offset point in height
        wOffset(0): how much to offset point in width
        **kwargs: standard layer keyword arguments
    """
    def __init__(self, crop_size, mean=0, std=1, hOffset=0, wOffset=0, **kwargs):
        super(PointCrop2D, self).__init__(**kwargs)
        self.cropSize = crop_size
        self.mean = mean
        self.std = std
        self.hOffset = hOffset
        self.wOffset = wOffset
        
    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `PointCrop2D` layer should be called on a list of two inputs')
        super(PointCrop2D, self).build(input_shape)
    
    def call(self, x):
        input_shape = [K.int_shape(x[0]), K.int_shape(x[1])]
        size = input_shape[1][1:]
        return K.map_fn(lambda i: _getCrop(x, size, self.cropSize, self.mean, self.std, self.wOffset, self.hOffset, i), K.arange(K.shape(x[1])[0]), dtype=K.floatx())
        
    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], self.cropSize, self.cropSize, input_shape[1][3])