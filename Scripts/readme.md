This folder contains the scripts used for preprocessing and analyzing H&E slide images.


|Script Name |Description|
|--|--|
|img_preprocessing.py |Tiles, filters, and stain normalizes/separates WSI and saves the images by subtype|
|transform_tile.py | Transforms tile images to artificially increase sample size for underrepresented subtypes |
|resnet_HE_v5.py | Trains deep learning model for H&E tile (256x256px) classification into one of 10 breast cancer subtypes. This is a transfer-learning ResNet-18 model adapted to our tile images.  Outputs information regarding accuracy and loss per epoch for training and validation, and reports final test set accuracies. |