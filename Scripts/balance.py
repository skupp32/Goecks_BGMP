#!/usr/bin/env python3

import os
import random
import shutil

# Define source and destination directories 
src_dirs = [
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/Basal",
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/HER2E",
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/LumA",
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/LumB",
    "/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_set/test/normal-like",
]

dest_dirs = [
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/Basal",
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/HER2E",
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/LumA",
    #"/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/LumB",
    "/projects/bgmp/shared/groups/2022/z7t/goecks/PAM50_balanced/test/normal-like",
]

#Select sample size
num_images = 6600

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