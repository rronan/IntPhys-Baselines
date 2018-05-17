IntPhys - Baseline
===============

Code accompanying the paper ["IntPhys: A Benchmark and Dataset for Intuitive Physics"](https://arxiv.org/abs/1803.07616)
This repository contains forward frame prediction models, which predict future images in synthetic videos. These predictions are either raw images or semantic masks.
At test time, models are used to predict if a video is physically plausible.

## Data

Training, dev and test sets can be found [here](http://www.intphys.com).
Training dataset consists in 15000 videos of 100 frames, showing object interactions and designed with UnrealEngine. It contains metadata (depth, masks, positions of objects...).
Dev dataset is made of 360 videos, half of which showing impossible events (e.g. an object disappearing). It contains metadata (depth, masks, positions of objects...), as well as the true label (`possible` or `impossible`).
Test dataset is made of 3600 videos, half of which showing impossible events (e.g. an object disappearing). It contains no metadata and no labels. One can submit predictions [here](http://www.intphys.com), a leaderboard will keep track challengers performances.

#### Train samples ####

Train samples are always physically possible and have high variability

<img src="examples/train_1.gif" width="150"> <img src="examples/train_2.gif" width="150"> <img src="examples/train_3.gif" width="150"> <img src="examples/train_4.gif" width="150">


#### Test and Dev samples ####

Test and Dev samples have a constrained variability and come as quadruplets: 2 possibles cases and 2 impossibles ones

<img src="examples/test_1.gif" width="150"> <img src="examples/test_2.gif" width="150"> <img src="examples/test_3.gif" width="150"> <img src="examples/test_4.gif" width="150">


#### Metadata ####

Each video comes with it's associated depth field and object masking
(each object have a unique id), along with a detailed status in JSON
format.

<img src="examples/meta_1.gif" width="150"> <img src="examples/meta_2.gif" width="150"> <img src="examples/meta_3.gif" width="150">

## Prerequisites

- Python 3.5
- [PyTorch](http://pytorch.org)
- Recommanded: NVIDIA GPU. CPU only is supported but very slow.
- Optional: [Visdom](https://github.com/facebookresearch/visdom) for visualization.

## Models

Each model is given an input sequence and a target sequence, specified by the options parameters ```--input_seq``` and ```--target_seq```. These parameters are patterns used by the dataloader to create inputs and targets. For example, from ```--input_seq 1 3 --target_seq 8 ``` the dataloader will return all triplets ```[1, 3 -> 8], [2, 4 -> 9], ..., [93, 95 -> 100]``` from every videos, in batches specified by option parameter ```--bsz```. 

Three models:

resnet_ae: pretrained resnet-18 followed by a deconvolution network:

gan: generative adversarial network as described in the paper.

linear_rnn: recurrent neural network applied to an encoded representation of a frame. This is a beta version, not presented in the paper.

## Dataloader

The dataloader uses a .npy lists containing absolute paths to all videos. Scripts ```makeList_train.py``` and ```makeList_test.py``` create those lists.

## Train

Train a mask predictor only:
```
python train.py --verbose --image_save --model Resnet_ae --input scene --target mask --input_seq 1 --target_seq 1
```

Train a forward model:
```
python train.py --verbose --image_save --model Resnet_ae --input scene --target mask --input_seq 1 3 --target_seq 8
```

Train a GAN model:
```
python train.py --verbose --image_save --input scene --model Gan --target mask --input_seq 1 3 --target_seq 8
```

Train a GAN on a predicted mask instead of a mask (so that test is done on raw videos):
```
python train.py --verbose --image_save --model Gan --input scene --target scene --input_seq 1 3 --target_seq 8 --maskPredictor path/to/trained/maskPredictor.pth
```

For GPU usage, add option ```--gpu```.
For visualization with [Visdom](https://github.com/facebookresearch/visdom), add option: ```--visdom```.

## A few notes

Given the size of the training set, one may want to save more than one checkpoint per epoch; this can be done with the option ```--n_slices``` (```--n_slices 3``` will save 3 checkpoints per epoch).


