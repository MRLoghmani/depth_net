from argparse import ArgumentParser
import numpy as np
import cv2
from os.path import join


def get_arguments():
    parser = ArgumentParser(
        description='Will convert 16bit grayscale images to colorject mapping as in \"Multimodal Deep Learning for Robust RGB-D Object Recognition\"')
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("image_list", help="File containing the relative path to each file we want to convert")

    args = parser.parse_args()
    return args

# Attention: the colorized depth image definition is: Red (close), blue(far)
# For TESTING pre-trained caffemodels use this function provided in this script 
# Note that the opposite definition can be found in function depth2jet.cpp blue(close), red(far)
# When re-training the network it should not make a difference, but the second definition, is necessary
# for noise augmentation. Nan values (which we convert to 0 distance) are therefore dark blue and close objects are also blue.
def scaleit_experimental(img):
    img_mask = (img == 0)
    istats = ( np.min(img[img>0]),	np.max(img))
    imrange=  1.0-(img.astype('float32')-istats[0])/(istats[1]-istats[0]);
    imrange[img_mask] = 0
    imrange= 255.0*imrange
    imsz = imrange.shape
    mxdim  = np.max(imsz)

    offs_col = (mxdim - imsz[1])/2
    offs_row = (mxdim - imsz[0])/2	
    nchan = 1
    if(len(imsz)==3):
        nchan = imsz[2]
    imgcanvas = np.zeros(  (mxdim,mxdim,nchan), dtype='uint8' )
    imgcanvas[offs_row:offs_row+imsz[0], offs_col:offs_col+imsz[1]] = img.reshape( (imsz[0],imsz[1],nchan) )
    # take rows
    if(offs_row):
        tr = img[0,:]
        br = img[-1,:]
        imgcanvas[0:offs_row,:,0] = np.tile(tr, (offs_row,1))
        imgcanvas[-offs_row-1:,:,0] = np.tile(br, (offs_row+1,1))
    # take cols
    if(offs_col):
        lc = img[:,0]
        rc = img[:,-1]
        imgcanvas[:, 0:offs_col,0] = np.tile(lc, (offs_col,1)).transpose()
        imgcanvas[:, -offs_col-1:,0] = np.tile(rc, (offs_col+1,1)).transpose()

    # RESCALE
    imrange_rescale = cv2.resize(imgcanvas, IMSIZE, interpolation=cv2.INTER_CUBIC) 
    return(imrange_rescale)


if __name__ == "__main__":
    args = get_arguments()
    output_dir =  args.output_dir
    input_dir = args.input_dir
    IMSIZE=(256,256)
    with open(args.image_list) as tmp:
        images = tmp.readlines()
    for i_path in images:
        #import pdb; pdb.set_trace()
        img_path = i_path.strip()
        img = cv2.imread(join(input_dir, img_path), -1);
        newimg = scaleit_experimental(img)
        newimg = cv2.applyColorMap(newimg, cv2.COLORMAP_JET)
        cv2.imwrite(join(output_dir, img_path), newimg)
