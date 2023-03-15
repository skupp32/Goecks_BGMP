#!/usr/bin/env python3

import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser(description = 'Select desired random sample from source directories and copy into balanced destination directories')
parser.add_argument('-sd','--src_dirs', required = True, nargs='+', help = 'List the source directories to pool files from')
parser.add_argument('-dd','--dest_dirs', required = True, nargs='+', help = 'List the directories you want to copy files to. These will be your balanced directories')
parser.add_argument('-s','--sample', required = True, nargs='+', help= 'What is your balanced sample size. This will be the minority class sample size')

args = parser.parse_args()

#Get the arguments (Source directories, destination directories, sample size)
src_dirs = args.src_dirs
dest_dirs = args.dest_dirs
num_images = args.sample

# Iterate over each source directory
for src_dir, dest_dir in zip(src_dirs, dest_dirs):
    # Get a list of all .png files in the source directory
    png_files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    # Select a random sample of tile files
    sample_files = random.sample(png_files, num_images)
    # Copy each file in the sample to the destination directory
    for file in sample_files:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copyfile(src_path, dest_path)
    print(f"Copied {num_images} random files from {src_dir} to {dest_dir}.")

#Below is an example of running this from command line:
#python balance.py -sd /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/Basal /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/HER2E /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/LumA /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/LumB /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/normal-like -dd /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/Basal /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/HER2E /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/LumA /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/LumB /projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/normal-like -s 6600