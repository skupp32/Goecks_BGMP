README for transform_tile.py script


## Overview:

This script performs horizontal and vertical flips to artificially upsample tiles.

### Argparse Arguments:

```
-i input directory
```
This option allows the user to specify the file path to where the tiles to be transformed are saved.

This is a required argument.

```
-o output directory
```

This option allows the user to specify the file path to thwere the transformed tiles should be saved.  As is, this script will not create the new file directories.

The default option for this argument is to save the tiles in the input directory.