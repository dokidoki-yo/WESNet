# WESNet(Weakly Supervised Extrinsics Self-Calibration of SVS)

This is the implementation of WESNet using PyTorch.

## Requirements

* CUDA
* PyTorch
* Other requirements  
    `pip install -r requirements.txt`

## Inference

* Image inference

    ```(shell)
    python inference.py --image_path $IMAGE_PATH$ --detector_weights $DETECTOR_WEIGHTS
    ```

## Dataset

 Download surround-view dataset from [here](https://cslinzhang.github.io/deepps/), and extract.

## Train

```(shell)
python train.py --dataset_directory $TRAIN_DIRECTORY --batch_size $BATCH_SIZE --enable_visdom
```

`TRAIN_DIRECTORY` is the train directory generated in data preparation.  
View `config.py` for more argument details (num_epochs, learning rate, etc).
