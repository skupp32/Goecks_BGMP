README for img_preprocessing.py script

## Overview:

This script takes a directory with Whole Slide Images (WSI) as .svs files, and preprocesses them for use in a deep learning algorithm.  The script as written is required to run on a gpu with cuda.

### Preprocessing Steps:

**Image tiling:**

We need to tile the image to increase the sample size and standardize iamge sizes.

After reading the .svs image into memory, the image_tiler() function breaks the image at a specified level into tiles of a specified size.

**Removing Whitespace:**

We need to remove whitespace to remove any tiles without stain as well as improve the performance of the stain normalization process.  We want to include only tiles that are mostly filled with stain.

This script uses a Gaussian Mixture Model to calculate a cutoff for average tile intensity.  The gmm_cutoff_calc() function calculates the average intensity of each tile. The mixture model then calculates peaks of intensity frequency which is used to find an upper and lower bound on tile intensity. 

*Intensity is defined as the average of the rgb values for all pixels in the tile.  

After calculating an intensity cutoff, the tile_filter_norm() function removes any tiles whose mean intensity is not between the bounds.

**Stain Normalization:**

We need to normalize the stain to remove batch effects resulting from different staining procedures.

This script uses the tile_filter_norm() function to stain normalize the tiles.  For this project, we used Macenko normalization as it is frequently cited.  It is also able to separate the stains to analyze only hematoxylin or eosin stains.


### Argparse Options:

```
-i input directory
```
This option allows the user to give the file path to a directory containg WSI (.svs).  

This is a required argument.


```
-l level
```
This option allows the user to specify what level of 'zoom' to view the WSI.  Values range between 0 and 19 where 19 is the highest resolution.  For this project we used level 15.  Higher levels will produce greater sample size at the cost of computation time.

This is a required argument.

```
-o outout directory
```
This option allows the user to give a directory to save tiles.  The script will create any directories that do not already exist, and will not overwrite any existing file structure.

The file structure will contain directories

<output_dir>/test/subtypes

<output_dir>/full_train/train/subtypes

<output_dir>/full_train/val/subtypes

The default argument is current working directory.

```
-c category
```
This option allows the user to select what kind of subtypes to separate images into. Options are PAM50, ANNOTATION, or both. Where PAM50 will separate into 5 PAM50 subtypes, ANNOTATION will separate into Invasive Ductal Carcinoma (IDC) and Invasive Lobular Carcinoma (ILC) and both will separate by both (10 total) types.

The default argument is PAM50.

```
-m manifest
```
This option allows the user to specify the file path to a manifest file.  This allows the script to associate image ids from the file name with subtype information from the manifest.

This a required argument.

```
-n number of images
```
This option  allows the user to specify how many WSI of each subtype to preprocess.  The value must be an integer.

The default argument is 5.


### Functions

**image_tiler(image_obj, level, tile_size)**

Reads in openslide object and creates tiles with specified size at the specified level.
Returns the tiles in a dictionary, and the number of tiles in every row and column

**tile_filter_norm(tiles,tile_array_shape,lower_bound, upper_bound, stain_type)**

Given an openslide image, desired level and tile size, compute the tiles, and remove tiles with average intensity below GMM cutoff.
Store in dictionary where the is the tile address and the value is the numpy array of the tile.
Returns dictionary and tuple with image size in number of tiles.
Has parameter to choose how to filter tiles.  If cutoff=mean, will keep tiles with less than mean intensity.  If cutoff is an integer
it will keep tiles with less than that integers percentile intensity.

**filtered_slide_reconstruction_plotter(filtered_tiles, tile_array_shape)**

Given the tiles in a dictionary and the shape of the tiles in the original image, this function will plot the tiles in their original position

**gmm_cutoff_calc(tiles,tile_array_shape,num_components)**

Given a dictionary with tiles, a tuple with the shape of the image in tiles, and the number of components to fit the histogram with 
(number of normal distributions) plot the histogram of mean tile intensities (sample of half the pixels) and return the value of the 
mean of the middle normal distribution to act as intensity cutoff.

**tile_saver(tiles,type_dict,output_dir,category,image_id)**
    
    Input:

    Dictionary where key is tile coordinate in entire slide and value is tile object

    Dictionary where key is slide id and value is (Histological Annotation, PAM50 Subtype)

    Output directory to put tiles

    Desired category for sorting

    Image filename from manifest

    Output:

    Saves tiles to file specified in function

    Given these inputs, the function saves the given tiles in the correct directories trying to achieve 70/20/10 train/val/test split.  Will not be exact as the test images need to be from distinct WSI from the train and val images.

    It will tile an image into train/val then test to ensure that there are tiles in each set, then split into the sets to achieve closest to the desired ratio.
