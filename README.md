<h1 align="center"> DPIPE</h1>

<p align="center">
  <img src="images/pipeguy.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="200" />
</p>

[![Documentation](http://inch-ci.org/github/dwyl/hapi-auth-jwt2.svg?branch=master)](https://multidataloader.readthedocs.io/en/latest/) 

With dpipe you can create ready to use datasets from paths or list of files. You should 
specify the type and the location of the input and target. The labels are assumed to be the name of the folder containing the file,
if you need a dataset for classification. 

The inputs and targets can be a list of paths, a path to be explored containing images or videos. For example:
````shell script
./dataset
  |
  |--cat/img1.png|
  |--cat/img2.png
  |--dog/img1.png
  |--dog/img2.png
````
The function `make_dataset` outputs a `dpipe.dataset_builder` object that has the method to predefined multiprocessing setups based on the recomendation of tensorflow.

| method        | action          
| ------------- |:-------------:| 
| `dataset_builder.prefetch()`      | Preloads samples on memory |
| `dataset_builder.batch()`      | Creates a batch dataset |
| `dataset_builder.enumerate()`      | Creates a appends an index to the output |
| `dataset_builder.filter()`      | Applies a filter concurrently |
| `dataset_builder.map()`      | Applies a function to each element concurrently |
| `dataset_builder.repeat()`      | Creates a repeated dataset |
| `dataset_builder.shuffle()`      | Shuffles the dataset after a complete run |
 

The dataset can be specified as:
````python
from dpipe import make_dataset
dataset = make_dataset('image','label',x_path='./dataset',x_size=(128,128)).build()
````
## Creating dataset (more options)
Additionally, we defined the dataset from functions or objects. Two use cases are presented here. A dataset can be created from a function and a list of element to parse, for example a list of files and a reading function. 
For example, if we need are training a denoising autoencoder, we need image noisy and clean image pairs; this can be handled with the function `dpipe.from_function`:
```python
import glob # to find the files
import matplotlib.image as mpimg # to read the images (you need to install it.)
import numpy as np
from dpipe import from_function

filelist = glob.glob('./dataset','*.png')
def read_file(filename):
    target = mpimg.imread(filename) # read the image
    noisy_image = np.random.randn(target.shape)
    return noisy_image, target
# undetermined shape is used to define dimentions that vary across shamples, in this case the height and the width of the images
dataset = from_function(read_file, filelist, undetermined_shape=((1,2),(1,2))).build()
```
If you are accessing your data in an object oriented way, you can use `dpipe.from_object`. In the next example lets consider you want use consume a list of files with records on it via generator function, this can also be handled with `dpipe.from_function` though. The code should look like this
```python
import os
import pandas as pd
from dpipe import from_object

class Reader():
    def __init__(self,datapath='./dataset'):
        self.filelist = os.listdir(datapath)
    def __len__(self):
        return len(self.filelist)
    def my_reading_function(self,filename):
        df = pd.read_csv(filename)
        for v, t in zip(df.values, df.targets):
            yield v, t
reader = Reader()
dataset = from_object(reader, 'my_reading_function','filelist').build()
```
The `build()` function that creates a dataset with arguments ready to use with the `fit()` method of and `tf.model` object. This is used like this:
```python
training_ds = from_object(reader_training, 'my_reading_function').shuffle(len(reader_training), reshuffle_each_iteration=True).batch(32).repeat().build()
validation_ds = from_object(reader_validation, 'my_reading_function',training=False).batch(32).build()
model.fit(x=training_ds,validation_data=validation_ds, epochs=10,**training_ds.built_args,**validation_ds_ds.built_args)
```
# Installation
````shell script
pip install dapipe
````
It requires to install FFMPEG ([here](https://www.ffmpeg.org)) to work with video formats.

