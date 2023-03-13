from PIL import Image
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="A program to transform pngs to artificially increase training set size.")
    parser.add_argument("-i","--input_directory",help = "The file path to an input png tiles to transform",required=True)
    parser.add_argument("-o","--output_directory", help = "The file directory of the output png for transformed tiles", default = 'input')
    return parser.parse_args()
	
args = get_args()
input_dir = args.input_directory
output_directory = args.output_directory

# Sets directory to save transformed tiles. If none given, save the tiles in place.
if output_directory == 'input':
    out_dir = input_dir
else:
    out_dir = output_directory


#loops over every tile in the directory
for tile in os.listdir(input_dir):

    #Transforms only original tiles (not tiles that have already been transformed)
    if tile.endswith('.png') and 'flipped' not in tile:

        #Reads in image with PIL
        img = Image.open(f'{input_dir}/{tile}')
        tile_name = tile.strip('.png')

        #Horizontally flips image and saves as png
        horiz_flip = img.transpose(method = Image.Transpose.FLIP_LEFT_RIGHT)
        horiz_flip.save(f'{out_dir}/{tile_name}_horiz_flipped.png')

        #Vertically flips image and saves as png
        vert_flip = img.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
        vert_flip.save(f'{out_dir}/{tile_name}_vert_flipped.png')

        #Diagonally flips image and saves as png
        both_flip = horiz_flip.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
        both_flip.save(f'{out_dir}/{tile_name}_vert_horiz_flipped.png')