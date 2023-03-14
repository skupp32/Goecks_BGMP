# Goecks_BGMP Repository

## Overview

This repositiory contains a collection of bash and python scripts to download, preprocess, and classify hematoxylin and eosin (H&E) whole slide image (WSI) breast cancer slides by clinical subtype. This project was contributed to by University of Oregon Knight Campus Bioinformatic and Genomics Master's Program students - Sam Kupp, Davin Marro, Sophia Soriano, Pranav Muthuraman - in collaboration with mentors at the OHSU Knight Cancer Institute Goecks Lab - Cameron Watson and Dr. Allison Creason.

## Repository Contents

This repository contains the following directories and files:
* Manifests: Contains the two initial The Cancer Genome Atlas (TCGA) manifest files - provided by OHSU project mentors 
    * final_TCGA_sample_manifest.txt: 875 TCGA H&E breast cancer slides with histological and clinical annotations for each image, along with unique slide IDs. Other breast cancer annotations are also included in this tab-separated file, but these are not used in this project.
    * GDC_download_manifest.txt: 875 TCGA image filenames corresponding to each of the 875 slide IDs in the above manifest file.
* Scripts: Contains the collection of scripts written for this project, along with accompanying README markdowns describing the usage script, and relevant background/supplementary information. Below is a table summarizing all scripts and READMEs in the order in which they are typically run:

|Script Name |Script README |
|--|--|
|[transform_tile.py](Scripts/transform_tile.py) |[README_transform_tile.md](Scripts/README_transform_tile.md) |
|[img_preprocessing.py](Scripts/img_preprocessing.py) |[README_img_preprocessing.md](Scripts/README_img_preprocessing.md) |
|[balance.py](Scripts/balance.py) |[README_balance.md](Scripts/README_balance.md) |
|[resnet_HE_v5.py](Scripts/resnet_HE_v5.py) |[README_resnet_HE_v5.md](Scripts/README_resnet_HE_v5.md) |
* Result_Images: Contains relevant result images generated over the course of the project, which are referenced and described in other markdowns throughout this repository.
* Goecks_BGMP_References.pdf: List of references consulted during this project.
