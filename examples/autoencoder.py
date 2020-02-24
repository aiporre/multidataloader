
import glob # to find the files
import matplotlib.image as mpimg # to read the images
import numpy as np
from dpipe import from_function

filelist = glob.glob('./dataset','*.png')
def read_file(filename):
    target = mpimg.imread(filename) # read the image
    noisy_image = np.random.randn(target.shape)
    return noisy_image, target
# undetermined shape is used to define dimentions that vary across shamples, in this case the height and the width of the images
dataset = from_function(read_file, filelist, undetermined_shape=((1,2),(1,2))).build()

