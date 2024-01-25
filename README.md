
## Requiments

- pytorch
- h5py
- yacs
- terminaltables
- tqdm
- transformers

## Start

### Data Preparation
We use the VGG feature provided by [2D-TAN](https://github.com/microsoft/VideoX) for the Charades-STA dataset, which can be downloaded from [here](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav/folder/137471415879). Please save it as `dataset/Charades-STA/vgg_rgb_features.hdf5`.
We use the C3D feature for the ActivityNet Captions dataset. Please download from [here](http://activity-net.org/challenges/2016/download.html) and save as `dataset/ActivityNet/sub_activitynet_v1-3.c3d.hdf5`. 


### Training
To train on the Charades-STA dataset:
```bash
sh scripts/charades_train.sh
```

To train on the ActivityNet Captions dataset:
```bash
sh scripts/anet_train.sh
```


### Inference

Run the following commands for evaluation:
```bash
sh scripts/eval.sh
```
