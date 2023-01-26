This folder contains the scripts used for preprocessing and analyzing H&E slide images.


|Script Name |Description|
|--|--|
|img_preprocessing.py |Tiles, filters, and stain normalizes/separates WSI and saves the images by subtype|
|transform_tile.py | Transforms tile images to artificially increase sample size for underrepresented subtypes |
|resnet_HE_v5.py | Trains deep learning model using ResNet-18 model and transfer learning from ImageNet.  Outputs information regarding training accuracy and loss and test accuracy |
