import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc
import time


def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)


def get_transformer(deploy_file, mean_file=None, mean_pixel=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
        inputs={'data': dims}
    )
    # transpose to (channels, height, width)
    t.set_transpose('data', (2, 0, 1))

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2, 1, 0))

    if mean_file:
        # set mean pixel
        print "Using mean file"
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(
                    blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError(
                    'blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            print pixel
            t.set_mean('data', pixel)
    if mean_pixel:
        print "Using mean pixel %f" % mean_pixel
        t.set_mean('data', np.ones(dims[1]) * mean_pixel)

    return t


def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
#    import pdb; pdb.set_trace()
    if image.mode == 'I':
        #print "impossible"
        data = np.array(image.resize([width, height], PIL.Image.BILINEAR))
        if mode == 'RGB':
            w, h = data.shape
            ret = np.empty((w, h, 3), dtype=np.float32)
            ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = data
            image = ret
        else:
            image = data.astype('float32')
    else:
        image = image.convert(mode)
        image = np.array(image)
        # squash
        image = scipy.misc.imresize(image, (height, width), 'bilinear').astype('float32')
    return image


def forward_pass(images, net, transformer,verbose, batch_size=None, layer_name='fc7'):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1
    batch_size = min(batch_size, len(images))
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:, :, np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    # fc6 = None
    fc7 = None
    todoChunks = [caffe_images[x:x + batch_size]
                  for x in xrange(0, len(caffe_images), batch_size)]
    start = time.clock()
    for k, chunk in enumerate(todoChunks):
#        if verbose == True:
#            print "Processing batch %d out of %d" % (k, len(todoChunks))
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            #import pdb; pdb.set_trace()
            net.blobs['data'].data[index] = image_data
        net.forward()
        if fc7 is None:
            # fc6 = net.blobs['fc6'].data
            fc7 = net.blobs[layer_name].data
        else:
            # fc6 = np.vstack((fc6, net.blobs['fc6'].data))
            fc7 = np.vstack((fc7, net.blobs[layer_name].data))
    print "It took %f" % (time.clock() - start)
    return fc7


class FeatureCreator:
    """This class keeps computed features in memory
    and returns them when requested"""

    def __init__(self, net_proto, net_weights, mean_pixel=None, mean_file=None, use_gpu=None, layer_name='fc7', verbose=False):
        self.net = get_net(net_weights, net_proto, use_gpu=use_gpu)
        self.transformer = get_transformer(
            net_proto, mean_pixel=mean_pixel, mean_file=mean_file)
        # self.features6 = {}
        self.features7 = {}
        self.layer_name = layer_name
        self.f_size = self.net.blobs[self.layer_name].data.shape[1]
        self.batch_size = 512
        self.scale = 1
        self.verbose = verbose

    def prepare_features(self, image_files):
        #import pdb; pdb.set_trace()
        _, channels, height, width = self.transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)
        images = [load_image(image_file, height, width, mode) for image_file in image_files]
	mean = 0.0
	for im in images:
	    mean += im.mean()
	mean = self.scale * mean / len(images)
	print "Image mean: %f" % mean
        if self.center_data:
            self.transformer.set_mean('data', np.ones(self.transformer.inputs['data'][1]) * mean)
            print "Will center data"
        # Classify the image
        feats = forward_pass(images, self.net,
                             self.transformer,self.verbose, batch_size=self.batch_size, layer_name=self.layer_name)
        i = 0
        # load the features in a map with their path as key
        for f in image_files:
            # self.features6[f] = fc6[i]
            self.features7[f] = feats[i]
            i += 1
        self.net = None  # free video memory

    def get_features(self, image_path):
        feats = self.features7.get(image_path, None)
        if feats is None:
            print "!!! Missing features for " + image_path
        return feats
      
    def set_data_scale(self, scale):
	if scale is None: return
	self.transformer.set_raw_scale('data', scale)
	print "Set transformer raw data scale to %f" % scale
	self.scale = scale
	
    def get_features_adv(self, image_files):
        fc6 = None
        fc7 = None
        for f in image_files:
            feats6 = self.features6[f]
            feats7 = self.features7[f]
            if fc6 is None:
                fc6 = feats6
                fc7 = feats7
            else:
                fc6 = np.vstack((fc6, feats6))
                fc7 = np.vstack((fc7, feats7))
        return (fc6, fc7)
