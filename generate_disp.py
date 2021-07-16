import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir",None,"Directory of kitti data",required=True)

feed_height = 192
feed_width = 640

ctx = mx.cpu(0)

model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_stereo_640x192',
                                    pretrained_base=False, ctx=ctx, pretrained=True)



def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def generate_disp(image):
    img = pil.open(image).convert('RGB')
    original_width, original_height = img.size
    img = img.resize((feed_width, feed_height), pil.LANCZOS)
    img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)
    outputs = model.predict(img)
    disp = outputs[("disp", 0)]
    disp_resized = mx.nd.contrib.BilinearResize2D(disp, height=original_height, width=original_width)
    disp_resized_np = disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy()
    scaled_disp, depth = disp_to_depth(disp_resized_np, 0.1, 100)
    return scaled_disp

def main(root_dir):
    os.makedirs(os.path.join(root_dir,'training/disparity'),exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'testing/disparity'), exist_ok=True)

    training_images = sorted(glob(os.path.join(root_dir,'training/image_2/*.png')))
    testing_images = sorted(glob(os.path.join(root_dir, 'testing/image_2/*.png')))



    for training_image in training_images:
        filename = training_image.split('/')[-1].split('.')[0]
        scaled_disp = generate_disp(training_image)
        np.save(scaled_disp,os.path.join(root_dir,'training/disparity/{}.npy'.format(filename)))

    for testing_image in testing_images:
        filename = testing_image.split('/')[-1].split('.')[0]
        scaled_disp = generate_disp(testing_image)
        np.save(scaled_disp,os.path.join(root_dir,'training/disparity/{}.npy'.format(filename)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.root_dir)
