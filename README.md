# hpml_project
# Simple 3D-GAN-PyTorch

<!-- [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/tf-3dgan/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)
 -->

## Introuction

* This is a very simple-to-use pytorch implementation of part of the [paper](https://arxiv.org/abs/1610.07584) "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling". I provide the complete pipeline of loading dataset, training, evaluation and visualization here and also I would share some results based on different parameter settings.


### Prerequisites

* Python 3.7.9 | Anaconda4.x
* Pytorch 1.6.0
* tensorboardX 2.1
* matplotlib 2.1
* visdom (optional)

### Pipeline

Basically I already put the `chair` dataset and a trained model as an example in `volumetric_data` and `outputs` folders. You can directly go to the training or evaluation part. But I still give a complete pipeline here.

#### Data

* We provide the chair dataset in `volumetric_data` folder from [ModelNet](http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip). As we use ModelNet instead of ShapeNet here, the results may be inconsistent with the paper.
* please unzip the `volumetric_data.zip` to use it for training and testing. 

#### Training
* Use `python main.py` to train the model.
* We have used different hyper_parameters to train and test the model and you can change the set of parameters in `param.py`. Please select `hyper_set` value between `1-8` to select the hyper-parameter set you want to train model. Please refer to the below table to know more abput the hyper_set.

|                             | Hyper_set | 1             | 2             | 3               | 4             | 5             | 6                   | 7                   | 8                   |
|-----------------------------|-----------|---------------|---------------|-----------------|---------------|---------------|---------------------|---------------------|---------------------|
| Parameters                  |           |               |               |                 |               |               |                     |                     |                     |
| Batch_size                  |           | 100           | 32            | 100             | 100           | 32            | 32                  | 32                  | 32                  |
| Generator Learning rate     |           | 0.0025        | 0.0020        | 0.0025          | 0.0025        | 0.0025        | 0.0025              | 0.0025              | 0.0025              |
| Discriminator Learning rate |           | 0.00001       | 0.00002       | 0.00001         | 0.00001       | 0.00001       | 0.00001             | 0.00001             | 0.00001             |
| Latent Vector Distribution  |           | Normal        | Normal        | Normal          | Normal        | Uniform       | Normal              | Normal              | Normal              |
| Generator Activation        |           | Sigmoid, ReLU | Sigmoid, ReLU | Tanh,ReLU       | Sigmoid, ReLU | Sigmoid, ReLU | Sigmoid,Leaky ReLU  | Sigmoid, ReLU       | Sigmoid, ReLU       |
| Discriminator Activation    |           | Sigmoid, ReLU | Sigmoid, ReLU | Tanh,Leaky ReLU | SIgmoid, ReLU | Sigmoid, ReLU | Sigmoid, Leaky ReLU | sigmoid, Leaky ReLU | Sigmoid, Leaky ReLU |
| Generator Optimizer         |           | Adam          | Adam          | Adam            | Adadelta      | Adam          | Adam                | SGD                 | Adam                |
| Discriminator Optimizer     |           | Adam          | Adam          | Adam            | Adadelta      | Adam          | Adam                | SGD                 | Adam                |
| Generator Loss              |           | L1Loss        | L1Loss        | L1Loss          | L1Loss        | L1Loss        | L1Loss              | L1Loss              | L1Loss              |
| Discriminator Loss          |           | MSELoss       | MSELoss       | MSELoss         | MSELoss       | MSELoss       | MSELoss             | MSELoss             | MSELoss             |

* During training, model weights and some 3D reconstruction images would be also logged to the `outputs` folders, respectively, for every `model_save_step` number of step in `params.py`.
* I have also attached the slurm job script `script_run.s`. You will have to make cetrain modification to it as per you HPC setting and schedule the job for training.

#### Evaluation
* For evaluation for trained model, you can run `python main.py --test=True` to call `tester.py`.
* If you want to visualize using visdom, first run `python -m visdom.server`, then `python main.py --test=True --use_visdom=True`.
* `results` folder containes training log and certain images of the 3D generated images with respect to each `hyper_parameter seeting from 1-9`. 


### Acknowledgements

* This code is a heavily modified version based on both [3DGAN-Pytorch](https://github.com/rimchang/3DGAN-Pytorch) and [tf-3dgan](https://github.com/meetshah1995/tf-3dgan) and thanks for them. Here I try to build a simpler but more complete pipeline, and explore more results with different settings as well.

