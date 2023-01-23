import matplotlib.pyplot as plt
import openslide
from openslide import deepzoom
from sklearn.mixture import GaussianMixture
import argparse
import cupy as cp
import numpy as np
from pathml.preprocessing import StainNormalizationHE
import seaborn
import os 
import random

def get_args():
    parser = argparse.ArgumentParser(description="A program to tile, filter, and stain normalize H&E slide images.  **Must be run on a node with a GPU**")
    parser.add_argument("-i","--input_dir",help = "The file path to an input svs image to preprocess",required=True)
    parser.add_argument("-l","--level",help = "The svs level to pick when examining slides",required=True)
    parser.add_argument("-o","--output_dir", help = "The file directory of the output png with preprocessed tiles", default = '.')
    parser.add_argument("-c","--category", help = 'Chooses PAM50 subtype or Histological Annotation for tile sorting post pre-processing.  Enter PAM50, ANNOTATION, or both',choices = ['PAM50','ANNOTATION','both'], default = 'PAM50')
    parser.add_argument("-m","--manifest", help = 'Path to manifest file to assosiate an image id with its classification', required = True)
    parser.add_argument("-n","--num_limit", help = "Number of tiles of each subtype to be processed.  Must be an integer", type = int)
    return parser.parse_args()
	
args = get_args()

img_dir = args.input_dir
level = int(args.level)
out_name = args.output_dir
cat = args.category
man_file = args.manifest
svs_lim = args.num_limit

print("** Saved argparse options **")


def image_tiler(image_obj, level: int, tile_size: int): #(dict,tuple):
    '''
    Reads in openslide object and creates tiles with specified size at the specified level.
    Returns the tiles in a dictionary, and the number of tiles in every row and column
    '''
    tiles = deepzoom.DeepZoomGenerator(image_obj, tile_size, overlap = 0, limit_bounds = True)
    num_tiles = tiles.level_tiles[level]

    tile_dict = {}

    for col in range(num_tiles[1]):
        for row in range(num_tiles[0]):
            tile_dict[(row,col)] = tiles.get_tile(level,(row,col))

    return (tile_dict,num_tiles)

def tile_filter_norm(tiles: dict,tile_array_shape: tuple,lower_bound: int, upper_bound: int, stain_type)-> dict:  # type: ignore
    '''
    Given an openslide image, desired level and tile size, compute the tiles, and remove tiles with average intensity below GMM cutoff.
    Store in dictionary where the is the tile address and the value is the numpy array of the tile.
    Returns dictionary and tuple with image size in number of tiles.
    Has parameter to choose how to filter tiles.  If cutoff=mean, will keep tiles with less than mean intensity.  If cutoff is an integer
    it will keep tiles with less than that integers percentile intensity.
    '''

    #Can be changed depending on process.  Here we just want to ensure that it is consistent across images
    method = "macenko"

    #Dictionary where key is row and column position of tile, and value will be the filtered, normalized slides
    filtered_tile_dict = {}

    # Calculate mean pixel intensity for each tile
    for col in range(tile_array_shape[1]):
        for row in range(tile_array_shape[0]):
            #Converts tiles to a cupy array
            tile_array = cp.array(tiles[(row,col)])
            #Calculates number of pixels based off array size accounting for all 3 channels
            num_pixels = cp.size(tile_array)

            #Randomly samples array and calculates mean to compare to cutoff
            cutoff_check = float(cp.mean(cp.random.choice(tile_array.ravel(),num_pixels//2,replace = False))) #.ravel() converts array to 1-D vector
            if cp.log1p(cutoff_check) < upper_bound and cp.log1p(cutoff_check) > lower_bound: 
                

                #Stain normalizes according to method used
                if stain_type == None:
                    filtered_tile_dict[(row,col)] = tile_array
                    continue
                if stain_type.lower() == 'all':
                    target = "normalize"
                elif stain_type.lower() == 'h':
                    target = "hematoxylin"
                elif stain_type.lower() == 'e':
                    target = "eosin"
                try:
                    normalizer = StainNormalizationHE(target = target, stain_estimation_method = method)
                    norm_tile = normalizer.F(tile_array.get())

                    filtered_tile_dict[(row,col)] = norm_tile #type: ignore
                except:
                    continue

    return filtered_tile_dict

def filtered_slide_reconstruction_plotter(filtered_tiles: dict, tile_array_shape: tuple)-> None:
    '''
    Given the tiles in a dictionary and the shape of the tiles in the original image, this function will plot the tiles in their original position
    '''
    #loop over every tile in the image
    for key,value in filtered_tiles.items():
        col = key[0]
        row = key[1]

        #assign tile subplot location in overall plot
        img_num = row*tile_array_shape[0] + col + 1
        plt.subplot(tile_array_shape[1],tile_array_shape[0],img_num)
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        for pos in ['right', 'top', 'bottom', 'left']:
            ax.spines[pos].set_visible(False)
        plt.imshow(cp.array(value).get())

    plt.subplots_adjust(wspace=0,hspace=0) #removes space between subplot images
    plt.suptitle(image_id)
    plt.savefig(f'preprocessed_test_images/hematoxylin/{image_id}_filtered_tiles.png')
    plt.close()

def gmm_cutoff_calc(tiles: dict,tile_array_shape,num_components: int)-> int:

    '''
    Given a dictionary with tiles, a tuple with the shape of the image in tiles, and the number of components to fit the histogram with 
    (number of normal distributions) plot the histogram of mean tile intensities (sample of half the pixels) and return the value of the 
    mean of the middle normal distribution to act as intensity cutoff.
    '''

    #Create array to store mean tile intensities in
    mean_array = cp.zeros((tile_array_shape[0],tile_array_shape[1]))

    for row in range(tile_array_shape[0]):
        for col in range(tile_array_shape[1]):
            tile_array = cp.array(tiles[(row,col)])
            if cp.shape(tile_array) != (256,256,3): #keeps only tiles of correct size- removes edge tiles
                continue
            num_pixels = cp.size(tile_array)
            mean_array[row,col] = float(cp.mean(cp.random.choice(tile_array.ravel(),num_pixels//2,replace = False))) #randomly samples tile to calculate mean intensity


    mean_array = mean_array[mean_array > 100]
    mean_array = cp.log1p(mean_array)

    gmm = GaussianMixture(n_components=num_components,covariance_type= 'full').fit(cp.reshape(mean_array.get(),(-1,1))) # type: ignore
    means = gmm.means_[:,0] #extracts means of gaussians
    i1, i2, i3 = cp.argsort(means)
    mean1, mean2,mean3 = means[[i1, i2, i3]]

    # calculate each component's standard deviations by taking sqrt of covariance
    std1, std2, std3 = gmm.covariances_[[i1, i2, i3],0,0] ** 0.5

    #Calculates intensity bounds for tiles to be included/removed from set
    #Can be adjusted to better include tiles     
    upper_bound = (mean1 + 2*std1)
    lower_bound = (mean1 - std1)

    fig, ax = plt.subplots()
    seaborn.kdeplot(mean_array.ravel().get(), ax = ax)

    plt.title(image_id)
    ax.set_xlabel('Log Mean Tile Intensity')
    plt.savefig(f"gmm_plots/log/{image_id}_gmm_fit.png")
    plt.close()

    return (lower_bound),(upper_bound)

def tile_saver(tiles:dict,type_dict:dict,output_dir:str,category: str,image_id: str)-> None:
    '''
    Input:
    Dictionary where key is tile coordinate in entire slide and value is tile object
    Dictionary where key is slide id and value is (Histological Annotation, PAM50 Subtype)
    Output directory to put tiles
    Desired category for sorting
    Image filename from manifest

    Output:
    Saves tiles to file specified in function
    '''
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    pam50 = type_dict[image_id][1]
    ann = type_dict[image_id][0]

    if category.lower() == 'pam50':
        img_class = pam50
    elif category.lower() == 'annotation':
        img_class = ann
    elif category.lower() == 'both':
        img_class = f'{ann}_{pam50}'
    
    print(f'{image_id}\t{type_dict[image_id]}\t{len(tiles)} remaining tiles.')

    num_train = len(os.listdir(f'{output_dir}/full_train/train/{img_class}'))
    num_val = len(os.listdir(f'{output_dir}/full_train/val/{img_class}'))
    num_test = len(os.listdir(f'{output_dir}/test/{img_class}'))


    print(f'{img_class}- train:{num_train}, validation:{num_val}, test:{num_test}')

    if num_train == 0:
        usage_set = 'train'
    elif num_test == 0:
        usage_set = 'test'
    else:
        if (num_train + num_val)/(num_test + num_train + num_val) <= 0.9:
            usage_set = 'train'
        else:
            usage_set = 'test'

    #loops over every tile in the whole slide image
    for coord, tile in tiles.items():
        row = coord[0]
        col = coord[1]

        #makes all saved tiles the same size
        plt.figure(figsize = (256*px, 256*px))
        plt.imshow(tile)
        ax = plt.gca()

        #Removes black outline and axes from image
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        for pos in ['right', 'top', 'bottom', 'left']:
                    ax.spines[pos].set_visible(False)

        #Sorts images into test/train/val probabilitistically
        # WILL NOT BE EXACT but should roughly converge to 70/20/10 split
        
        if usage_set.lower() == 'train':
            rand_num = float(cp.random.rand(1))
            if rand_num < 0.7*10/9:
                set = 'train'
            else:
                set = 'val'
        elif usage_set.lower() == 'test':
            set = 'test'

        #saves images in specified outfile directory according to image classification in format <image_id>_<annotation>_<PAM50 subtype>_<row>_<col>.png
        if set == 'train' or set == 'val':
            plt.savefig(f'{output_dir}/full_train/{set}/{img_class}/{image_id}_{type_dict[image_id][0]}_{type_dict[image_id][1]}_{row}_{col}.png',bbox_inches = 'tight',pad_inches = 0)
        elif set == 'test':
            plt.savefig(f'{output_dir}/{set}/{img_class}/{image_id}_{type_dict[image_id][0]}_{type_dict[image_id][1]}_{row}_{col}.png',bbox_inches = 'tight',pad_inches = 0)
        
        plt.close()


#Ensures that the file saving commands are correct format
if out_name.endswith('/'):
    out_name = out_name[:-1]

img_type_dict = {} #Dictionary where the slide filename from manifest is the key and (Histological Annotation, PAM50 Subtype) is the value


#Creates folders to save images in depending on type of classification
sets = ['train','val','test']

if cat.lower() == 'pam50':
    for set in sets:
        for type in ['Basal','LumA','LumB','normal-like','HER2E']:

            try:
                if set == 'train' or set == 'val':
                    os.makedirs(f'{out_name}/full_train/{set}/{type}')
                elif set == 'test':
                    os.makedirs(f'{out_name}/{set}/{type}')
            except FileExistsError:
                continue

elif cat.lower() == 'annotation':
    for set in sets:
        for type in ['Invasive_lobular_carcinoma','Invasive_ductal_carcinoma']:

            try:
                if set == 'train' or set == 'val':
                    os.makedirs(f'{out_name}/full_train/{set}/{type}')
                elif set == 'test':
                    os.makedirs(f'{out_name}/{set}/{type}')
            except FileExistsError:
                continue

elif cat.lower() == 'both':
    for set in sets:
        for type1 in ['Basal','LumA','LumB','normal-like','HER2E']:
            for type2 in ['Invasive_lobular_carcinoma','Invasive_ductal_carcinoma']:
                type = f'{type2}_{type1}'
                
                try:
                    if set == 'train' or set == 'val':
                        os.makedirs(f'{out_name}/full_train/{set}/{type}')
                    elif set == 'test':
                        os.makedirs(f'{out_name}/{set}/{type}')                    
                except FileExistsError:
                    continue

print("** Files Created **")


# Read in information from manifest file with info on subtype and annotation to label images
with open(man_file,'r') as man:
    for line in man:
        if line.startswith('TCGA'):
            line = line.strip()
            line = line.split('\t')
            pam50 = line[2].replace(' ','_')
            hist_ann = line[1].replace(' ','_')
            img_filename = line[9][:-4]
            img_type_dict[img_filename] = (hist_ann,pam50)
#Dictionary where key is subtype classification and value is the number of slides that have been processed of that type
print("** Manifest File Parsed **")

wsi_type_count = {}
file_list = os.listdir(img_dir)
random.shuffle(file_list)

for img_name in file_list:
    print(img_name)
    if not img_name.endswith('.svs'):
        continue
    # Grabs the id from the image name to identify WSI
    image_id = img_name.split('/')[-1][:-4]
    #open svs image
    svs_img = openslide.open_slide(f'{img_dir}/{img_name}')

    #Grab type info from dictionary with info from manifest file
    pam_50 = img_type_dict[image_id][1]
    hist_ann = img_type_dict[image_id][0]

    #Create variable containing subtype info
    if cat.lower() == 'both':
        subtype = f'{hist_ann}-{pam_50}'
    elif cat.lower() == 'pam50':
        subtype = pam_50
    elif cat.lower() == 'annotation':
        subtype = hist_ann
    
    #Ensures that only the desired number of WSI are processed
    if subtype not in wsi_type_count:
        wsi_type_count[subtype] = 1
    else:
        if wsi_type_count[subtype] < svs_lim:
            wsi_type_count[subtype] += 1
        else:
            continue


    #Actual preprocessing steps

    tile_dictionary, number_tiles = image_tiler(svs_img,level,256)
    print('** Tiling Complete **')

    lower_bound, upper_bound = gmm_cutoff_calc(tile_dictionary, number_tiles,3)
    print('** Intensity Cutoff Calculated **')

    filtered_norm_tile_dict = tile_filter_norm(tile_dictionary,number_tiles,lower_bound,upper_bound, 'h')
    print('** Tiles Filtered and Normalized **')

    if float(cp.random.rand(1)) < 1/10:
        filtered_slide_reconstruction_plotter(filtered_norm_tile_dict, number_tiles)
        print('** Whole Slide Image Reconstructed **')

    tile_saver(filtered_norm_tile_dict, img_type_dict, out_name, cat, image_id)
    print('** Hematoxylin Tiles Saved **')

    filtered_norm_tile_dict = tile_filter_norm(tile_dictionary,number_tiles,lower_bound,upper_bound, 'all')
    print('** Tiles Filtered and Normalized **')

    tile_saver(filtered_norm_tile_dict, img_type_dict, '/projects/bgmp/shared/groups/2022/z7t/goecks/hematoxylin_comp_set/', cat, image_id)
    print('** Normalized Tiles Saved **')

print(wsi_type_count)