# dgcn

This is an implementation of [Dual Graph Convolutional Network for Semantic Segmentation](https://arxiv.org/abs/1909.06121) for my graduate school assignment.   Because this is not an official implementation, I can't guarantee the implementation correctness, performance of model, or any feedback through issues.

* * *

## How to use

1. Download [Cityscapes](https://www.cityscapes-dataset.com/) dataset. Both raw images and semantic images should be downloaded.
2. Create `/cityscapes` directory and unzip downloaded dataset on that directory.
3. Create `/model` directory to locate weight files.
4. To train,
```console
foo@bar:~/dgcn$ python3 train.py
```
5. To inference,
```console
foo@bar:~/dgcn$ python3 infer.py aachen_000000_000019
```
I am not accessible to the test set of Cityscapes, thus the `infer.py` does not inference images in the test set. You need to modify the script if you need to inference them.
6. To test,
```console
foo@bar:~/dgcn$ python3 test.py
```
The script calculates IoU of each class and MIoU of validation set.


Pretrained weight is trained for 180 epochs.
[Weight Download Link](https://drive.google.com/file/d/1XAHEbx1xZJTMmG2Ha2LvpyCa59R5RMmj/view?usp=sharing)
