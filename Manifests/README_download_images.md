README to describe how to access data from TCGA

## Overview

Images came from The Cancer Genome Atlas and were accessed via a manifest file.  The manifest file we used to download the images for this project is [GDC_download_manifest.txt](https://github.com/skupp32/Goecks_BGMP/blob/main/Manifests/GDC_download_manifest.txt) in the Manifests folder.

On our HPC, Talapas, we downloaded images with the commands:

```
module load racs-eb/1
module load gdc-client/1.3.0-intel-2017b-Python-2.7.14

gdc-client download -m ./GDC_download_manifest.txt
```

The `module load` commands load the proper packages for gdc-client.

As loaded, the packages have requirements for processor architecture to run.  According to the documentation any of the following architectures will work: `Intel(R) X87, CMOV, MMX, FXSAVE, SSE, SSE2, SSE3, SSSE3, SSE4_1, SSE4_2, MOVBE, POPCNT, F16C, AVX, FMA, BMI, LZCNT and AVX2`.  

We used an Intel processor for downloading.