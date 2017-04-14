from __future__ import division
import caffe
import numpy as np
from scipy import ndimage 
import random
from random import randint
from multiprocessing import Process

class DoItAll(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1,            'requires a single layer.bottom'
        assert bottom[0].data.ndim >= 3,    'requires image data'
        assert len(top) == 1,               'requires a single layer.top'

    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        fill=0
        # Then zero-out one fourth of the image
        height = 227
        width = 227
        obj_mean = bottom[0].data[bottom[0].data<255.0].mean()
        h_offset = random.randrange(int(height / 2 + height / 4))
        w_offset = random.randrange(int(width / 2 + width / 4))
        bottom[0].data[...,
                    h_offset:(h_offset + height / 4),
                    w_offset:(w_offset + width / 4),
                    ] = fill
        scale = random.uniform(0.5,2.0)
        offset = random.uniform(-20,20)
        shift = -128.0#-128.0 ##this is to center the images; we chose 128 as the image is from 0 to 255 (but usually its real mean is more than 128)
        def subf(bottom,top,start,finish):
            for i in range(start,finish): #just to remember:it will stop looping with last variable finish-1 !
                Bx = random.uniform(0.0,0.5) # amount of shearing in x-axis
                By = random.uniform(0.0,0.5) # amount of shearing in y-axis
                sign=randint(0,1)
                if sign==0:
                    Bx = Bx * (-1)
                    By = By * (-1)
                shear_array=np.array([[1,Bx],[By,1]])
                top[0].data[i,0,:] = ndimage.affine_transform( bottom[0].data[i,0,:],matrix=[[1,Bx],[By,1]],order=0,cval=255.0)

	batch_size=bottom[0].data.shape[0]
	divided_batch=batch_size // 4 #integer division, from the future!
	if False: #divided_batch > 1:
	    p1 = Process(target=subf, args=(bottom,top,0,divided_batch,))
	    p2 = Process(target=subf, args=(bottom,top,divided_batch,divided_batch*2,))
            p3 = Process(target=subf, args=(bottom,top,divided_batch*2,divided_batch*3,))
            p4 = Process(target=subf, args=(bottom,top,divided_batch*3,batch_size,))

	    p1.start()
	    p2.start()
            p3.start()
            p4.start()
            p1.join()
            p2.join()
	    p3.join()
	    p4.join()
	else:#It cannot be parallelizable, or it is just batch size = 1 (for example classify one of DIGITS)
	    subf(bottom,top,0,batch_size)      
        bg = (top[0].data==255.0)
        top[0].data[bg] = np.random.uniform(obj_mean, 255.0)  # between the object mean and 255
        NOISE=5
        top[0].data[...] = (top[0].data + shift) * scale + offset + np.random.uniform(-NOISE, NOISE, top[0].data.shape) # also adds noise

    def backward(self, top, propagate_down, bottom):
        pass

