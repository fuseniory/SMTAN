## Scene-Enhanced Multi-Scale Temporal Aware Network for Video Moment Retrieval

## Prerequisites

```
pytorch transformers yacs h5py terminaltables tqdm  
```

## Datasets

For the video features, we use VGG features for Charades-STA, and C3D features on ActivityNet Captions. Download the corresponding video features provided by [2D-TAN](https://github.com/microsoft/2D-TAN). 

## Train

Run the following command to train on the Charades-STA dataset:
```bash
sh scripts/charades_train.sh
```

Run the following command to train on the ActivityNet Captions dataset:
```bash
sh scripts/anet_train.sh
```


## Inference

Run the following command for evaluation:
```bash
sh scripts/eval.sh
```

Our trained model are provided in [GoogleDrive](https://drive.google.com/drive/folders/1Mdw8oG0cPwr5lvvjCawzjIUKByhFY3hS?usp=sharing). Please download them to the `outputs` folder.
