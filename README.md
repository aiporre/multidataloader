<h1 align="center"> DPIPE</h1>

<p align="center">
  <img src="images/pipeguy.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="200" />
</p>

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
The dataset can be specified as:
````python
from dpipe import make_dataset
dataset = make_dataset('image', 'label', x_path='./dataset', x_size=(128,128)).build()
````
