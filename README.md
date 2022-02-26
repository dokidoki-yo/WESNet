# WESNet(Weakly Supervised Extrinsics Self-Calibration of SVS)

This is the implementation of WESNet using PyTorch. Here is our [paper](https://ieeexplore.ieee.org/document/9693241).
![]()

## Requirements

For training:
* CUDA
* PyTorch
* * Other requirements (such as visiualization of loss curve)
    `pip install -r requirements.txt`

For data preprocessing:
* OpenCV
* Eigen
    
## Dataset Download

 Download our surround-view dataset containing original fisheye images from [here] (password:cblf) (https://pan.baidu.com/s/1BxdyM30Nysq7NABBML_Kgg#list/path=%2F), and extract. The labels of the calibration site images consisting of the 2D pixel coordinates and corresponding 3D world coordinates of the selected corners are also provided in 'C_f.zip'.

## Data Preparation
* Labels: Before training, make sure that each lable file contains the same number of point pairs, we suggest keeping at least 15 pairs.
* Augmentation: Add the disturbance within a certain range to the original dataset for data augmentation. To prove efficiency, we implemented it in C++ (see prepare_training.cpp), and don't forget to change the directory to your own directory when reading and saving images.

## Train

```(shell)
python train.py --dataset_directory $TRAIN_DIRECTORY --batch_size $BATCH_SIZE --enable_visdom
```

`TRAIN_DIRECTORY` is the directory of training data generated in data preparation.  
`--enable_visdom` is to help visiualize the loss curve in the training process.

View `config.py` for more argument details (num_epochs, learning rate, etc).

## Inference

```(shell)
python inference.py --image_path $IMAGE_PATH$ --detector_weights $DETECTOR_WEIGHTS
```
`IMAGE_PATH` is the directory of testing data generated in data preparation, in which the organization of folders and names of testing images is the same as the training set.
`DETECTOR_WEIGHTS` is the model of trained weights.

