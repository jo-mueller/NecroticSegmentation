# Necrotic Segmentation
Methods to segment necrotic tissue from typical HE images. More example image data and usage notes will follow!

## Training:
```--> Train.py```
Training can be done based on annotated tiles which can generated, for instance, with QuPath as described [here](https://qupath.readthedocs.io/en/latest/docs/advanced/exporting_images.html). A modified script to export tiles with names as expected by Necrotic Segmentation is given by ```Utils/tilify_annotations.groovy```

Annotations can be given in a one-hot encoding standard or with exlusive labels. Images and annotations should be in the same directory and should be named in the following fashion:

```
\dir
  SomeImageTile_xy.tif
  SomeImageTile_xy-labelled.tif
```

### Settings

Adjust the following parameters according to your need. More example data will follow.
```
# Config
src = root + r'\YourTileDataDirectory'
BATCH_SIZE =28  # batch size- tweak and see how mch fits on your GPU, depends on tile size
USE_SAMPLER = False  
SAMPLER  = None
num_workers = 0  
LEARNING_RATE = 2e-5  # learning rate - controls how fast the networks converges
criterion = CrossEntropyLoss()  # loss criterion
EPOCHS = 200  # number of epochs for training
IMG_SIZE = int(os.path.basename(src).split('_')[1])  # image size will be inferred from directory name
PIX_SIZE = float(os.path.basename(src).split('_')[2])  # pix size will be inferred from directory name
PATIENCE = 20  # training will stop if no erformance improvement was found after this number of epochs
device= 'cuda'
keep_checkpoints = True
```

## Inference
```--> Process.py``` This script allows running the net in inference mode, all you need is a dataframe (list of filenames) of czi-images to be processed, which can then be passed to the network. The output will be saved in a path relative to the input image:
```
InputImage.czi
-->\1_seg\'HE_seg_DL.tif'
```
For inference, czi image will be loaded in the same resolution as used for training. The image will then be split up in tiles (same size as training) and fed forward through the net. This process will be repeated a number of times and the tiles will be offset by a set value in each iteration to avoid edge effects.

### Settings
The following parameters can/should be adjusted for inference:

```

root = r'E:\YourRepositoryRootDir'
RAW_DIR = r'E:\Promotion\Projects\2020_Radiomics\Data'  # root dir that coontains to-be-processed H&E images
EXP = root + r'\data\Experiment_20210426_110720'  # experiment pat that contains model file (amoong others)
DEVICE = 'cuda'
STRIDE = 16  # Tile-edge that will be cropped away before puzzling the image back together
Redo = True  # only important if you run segmentation multiple times on the same data
MAX_OFFSET = 64  # offset between subsequent tiles
N_OFFSETS = 10  # number of used tile ooffsets
SERIES = 2  # most suitable magnification level of czi data
