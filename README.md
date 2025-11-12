# ELEC475_Lab3

pip install -r requirements.txt

### Dataset Setup
Download the Pascal VOC 2012 dataset manually:
1. Download dataset from https://www.kaggle.com/datasets/bardiaardakanian/voc0712.
2. Extract the contents from the archive folder into `./data/VOCdevkit/VOC2012/`

### Data Directory Structure
After extraction, your data folder should look like this:
```
data/
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/          # XML annotation files
        ├── ImageSets/
        │   ├── Action/
        │   ├── Layout/
        │   ├── Main/             # Train/val/test splits
        │   └── Segmentation/
        ├── JPEGImages/           # Original images
        ├── SegmentationClass/    # Segmentation masks (class)
        └── SegmentationObject/   # Segmentation masks (object)
```

