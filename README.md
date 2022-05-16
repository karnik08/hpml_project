# hpml_project
# Simple 3D-GAN-PyTorch

## Introuction

* This is a pytorch implementation of the [paper](https://arxiv.org/abs/1610.07584)"Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling"that is very easy to use. I'll show you the entire process from importing the dataset to training, evaluation, and visualization, as well as sample results based on various parameter values.


### Prerequisites

* Python 3.7.9 | Anaconda4.x
* Pytorch 1.6.0
* tensorboardX 2.1
* matplotlib 2.1
* visdom (optional)

### Pipeline

I just put the `chair` dataset and a trained model in the `volumetric data`. When you train the model `outputs` folder will be created and you can see the logs and out images in it. You can skip right to the training or evaluation sections. However, I provide a whole pipeline here.

#### Data

* We provide the chair dataset in `volumetric_data` folder from [ModelNet](http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip). As we use ModelNet instead of ShapeNet here, the results may be inconsistent with the paper.
* Download the data from [data](https://drive.google.com/file/d/1azxHE830qjntfrwGxSoZZhavnXio99mZ/view?usp=sharing)
* please unzip the `volumetric_data.zip` to use it for training and testing. 

#### Training
* Use `python main.py` to train the model.
* We have used different hyper_parameters to train and test the model and you can change the set of parameters in `param.py`. Please select `hyper_set` value between `1-8` to select the hyper-parameter set you want to train model. Please refer to the below table to know more abput the hyper_set.

|                             | Hyper_set | 1             | 2             | 3               | 4             | 5             | 6                   | 7                   | 8                   |
|-----------------------------|-----------|---------------|---------------|-----------------|---------------|---------------|---------------------|---------------------|---------------------|
| Parameters                  |           |               |               |                 |               |               |                     |                     |                     |
| Batch_size                  |           | 100           | 32            | 100             | 100           | 32            | 32                  | 32                  | 32                  |
| Generator Learning rate     |           | 0.0025        | 0.0020        | 0.0025          | 0.0025        | 0.0025        | 0.0025              | 0.0025              | 0.0025              |
| Discriminator Learning rate |           | 0.00001       | 0.000005       | 0.00001         | 0.00001       | 0.00001       | 0.00001             | 0.00001             | 0.00001             |
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
* `results` folder containes training log and certain images of the 3D generated images with respect to each `hyper_parameter seeting from 1-9`. 


### Acknowledgements

 * [3DGAN-Pytorch](https://github.com/rimchang/3DGAN-Pytorch)
 * [tf-3dgan](https://github.com/meetshah1995/tf-3dgan) 
 * [pytorch-3dgan](https://github.com/xchhuang/simple-pytorch-3dgan)


