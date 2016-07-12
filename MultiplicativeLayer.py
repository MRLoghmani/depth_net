import caffe
import numpy as np
import random
from random import randint
import sunpy.image.transform


class Multiplicative(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 1,            'requires a single layer.bottom'
        assert bottom[0].data.ndim >= 3,    'requires image data'
        assert len(top) == 1,               'requires a single layer.top'

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def _bak_forward(self, bottom, top):
        # Copy all of the data
        app = bottom[0].data[...]
        # Take a number
        scale = np.random.uniform(0.5, 2)
        offset = np.random.uniform(-20, 20)
        # Scale the values
        top[0].data[...] = app * scale + offset

    def forward(self, bottom, top):
        app = bottom[0].data[...]
        batch_size = app.shape[1]
        for i in range(0, batch_size):
            # (rows,cols)=np.shape(app)
            img = app[i]
            Bx = 0.5#random.uniform(0.0, 0.5)  # amount of shearing in x-axis
            By = 0.5#random.uniform(0.0, 0.5)  # amount of shearing in y-axis
            sign = randint(0, 1)
            if sign == 0:
                Bx = Bx * (-1)
                By = By * (-1)
            shear_array = np.array([[1, Bx], [By, 1]])
            dst = sunpy.image.transform.affine_transform(
                img, shear_array, image_center=(118, 118), missing=127.0)
            top[0].data[...][i] = dst
        scale = np.random.uniform(0.5, 2.0)
        offset = np.random.uniform(-20, 20)
        # Scale the values
        top[0].data[...] = top[0].data[...] * scale + offset

    def backward(self, top, propagate_down, bottom):
        pass


class BlankSquareLayer(caffe.Layer):

    def setup(self, bottom, top):
        assert len(bottom) == 1,            'requires a single layer.bottom'
        assert bottom[0].data.ndim >= 3,    'requires image data'
        assert len(top) == 1,               'requires a single layer.top'

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        # Copy all of the data
        top[0].data[...] = bottom[0].data[...]
        # Then zero-out one fourth of the image
        height = top[0].data.shape[-2]
        width = top[0].data.shape[-1]
        h_offset = random.randrange(height / 2 + height / 4)
        w_offset = random.randrange(width / 2 + width / 4)
        top[0].data[...,
                    h_offset:(h_offset + height / 4),
                    w_offset:(w_offset + width / 4),
                    ] = 0

    def backward(self, top, propagate_down, bottom):
        pass
