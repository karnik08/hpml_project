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

[comment]: <> (* First, click [here]&#40;http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip&#41; to download the dataset. Then unzip it and put the `volumetric_data` folder to the path of our main repository. As we use ModelNet instead of ShapeNet here, the results may be inconsistent with the paper.)
* We provide the chair dataset in `volumetric_data` folder from [ModelNet](http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip). As we use ModelNet instead of ShapeNet here, the results may be inconsistent with the paper.

#### Training
* Use `python main.py` to train the model.
* We have used different hyper_parameters to train and test the model and you can change the set of parameters in `param.py`. Please select `hyper_set` value between `1-9` to select the hyper-parameter set you want to train model.
* During training, model weights and some 3D reconstruction images would be also logged to the `outputs` folders, respectively, for every `model_save_step` number of step in `params.py`.

#### Evaluation
* For evaluation for trained model, you can run `python main.py --test=True` to call `tester.py`.
* If you want to visualize using visdom, first run `python -m visdom.server`, then `python main.py --test=True --use_visdom=True`.


### Acknowledgements

* This code is a heavily modified version based on both [3DGAN-Pytorch](https://github.com/rimchang/3DGAN-Pytorch) and [tf-3dgan](https://github.com/meetshah1995/tf-3dgan) and thanks for them. Here I try to build a simpler but more complete pipeline, and explore more results with different settings as well.

