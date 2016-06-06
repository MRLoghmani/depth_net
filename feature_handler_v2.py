import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc
import time
from tqdm import tqdm

def get_net(caffemodel, deploy_file, use_gpu=True, GPU_ID=0):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        print "Using GPU %d" % GPU_ID
        caffe.set_device(GPU_ID)
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
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
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
    image = image.resize((width, height), PIL.Image.BILINEAR)
    if image.mode == 'I':
        tmp = np.array(image)
        if mode == 'RGB':
            w, h = tmp.shape
            data = np.empty((w, h, 3), dtype=np.float32)
            data[:, :, 2] = data[:, :, 1] = data[:, :, 0] = tmp
        else:
            data = tmp
    else:
        image = image.convert(mode)
        data = np.array(image)
    return data


def forward_pass(images, net, transformer, batch_size=1, layer_name='fc7'):
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
#    import ipdb; ipdb.set_trace()
    batch_size = min(batch_size, len(images))
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:, :, np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]
    fsize = net.blobs[layer_name].data.shape[1]
    features = np.empty((len(images), fsize), dtype='float32') 
    todoChunks = [caffe_images[x:x + batch_size]
                  for x in xrange(0, len(caffe_images), batch_size)]
    start = time.clock()
    idx = 0
    for k, chunk in tqdm(list(enumerate(todoChunks))):
        bsize = len(chunk)
        new_shape = (bsize,) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        net.forward()
        features[idx:idx + bsize] = np.squeeze(net.blobs[layer_name].data.copy())
        idx += bsize
    print "It took %f" % (time.clock() - start)
    return features


class FeatureCreator:
    """This class keeps computed features in memory
    and returns them when requested"""

    def __init__(self, net_proto, net_weights, mean_pixel=None, mean_file=None, use_gpu=True, layer_name='fc7', verbose=False, gpu_id=0):
        self.net = get_net(net_weights, net_proto, use_gpu, gpu_id)
        self.transformer = get_transformer(
            net_proto, mean_pixel=mean_pixel, mean_file=mean_file)
        # self.features6 = {}
        self.features = {}
        self.layer_name = layer_name
        self.f_size = self.net.blobs[self.layer_name].data.shape[1]
        self.batch_size = 256
        self.scale = 1
        self.verbose = verbose
        self.data_prefix = ''

    def prepare_features(self, image_files):
        #import ipdb; ipdb.set_trace()
        old_batch_size, channels, height, width = self.transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)
        print "Loading images"
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
        print "Extracting features"
        feats = forward_pass(images, self.net,  self.transformer, batch_size=self.batch_size, layer_name=self.layer_name)
        i = 0
        # load the features in a map with their path as key
        for f in image_files:
            short_name = f.replace(self.data_prefix, '')  # saves only the relative path
            if short_name[0] == '/':
                short_name = short_name[1:]
            self.features[short_name] = feats[i].reshape(feats[i].size)
            i += 1
        self.net = None  # free video memory

    def get_features(self, image_path):
        feats = self.features.get(image_path, None)
        if feats is None:
            print "!!! Missing features for " + image_path
        return feats
      
    def set_data_scale(self, scale):
	if scale is None: return
	self.transformer.set_raw_scale('data', scale)
	print "Set transformer raw data scale to %f" % scale
	self.scale = scale

    def do_forward_pass(self, data):
        return forward_pass(data, self.net, self.transformer,self.verbose, batch_size=self.batch_size, layer_name=self.layer_name)
