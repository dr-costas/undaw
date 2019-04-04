# UNDAW Repository

### Welcome to the repository of UNDAW (Unsupervised Adversarial Domain Adaptation Based on the Waserstein Distance)

This is the repository for the method presented in the paper: 
"Unsupervised Adversarial Domain Adaptation Based on the Waserstein Distance," 
by [K.
Drossos](https://tutcris.tut.fi/portal/en/persons/konstantinos-drosos(b1070370-5156-4280-b354-6291618bb965).html), 
[P. Magron](http://www.cs.tut.fi/~magron/), and [T. Virtanen](http://www.cs.tut.fi/~tuomasv/). 

Our paper is submitted for review at the [2019 IEEE Workshop on Applications of Signal Processing to Audio 
and Acoustics (WASPAA), Mohonk Mountain House, New Paltz, NY](https://www.waspaa.com/). 

You can find an online version of our paper at arxiv:    

**If you use our method, please cite our paper**: 

---

## Table of Contents

1. [ Dependencies, pre-requisites, and setting up the code ](#dependencies-pre-requisites-and-setting-up-the-code)
2. [ Reproduce the results of the paper ](#reproduce-the-results-of-the-paper)
3. [ Use the code with your own data ](#use-the-code-with-your-own-data)
4. [ Previous work ](#previous-work)
5. [ Acknowledgement ](#acknowledgement)

---

## Dependencies, pre-requisites, and setting up the code
 
In order to use our code, you have to firstly:
 
* use Python 3.x and install all the required packages listed at the
[requirements file for PiP](https://github.com/dr-costas/undaw//blob/master/requirements.txt) or at the
[requirements file for Anaconda](https://github.com/dr-costas/undaw//blob/master/conda_requirements.txt) 
* download the data (the file ``AUDASC_features_labels.zip``) from 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1164585.svg)](https://zenodo.org/record/1401995#.W31Zaxx9iK4)
* download the pre-trained non-adapted model (the file ``AUDASC_pretrained_weights.zip``) from 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1164585.svg)](https://zenodo.org/record/1401995#.W31Zaxx9iK4) 
and the adapted model from 
 
(this is optional and is required only in the case that you want to reproduce the 
results of the paper)

Then: 
* unzip the file ``AUDASC_features_labels.zip``. This will produce the following files, which
will have to be placed inside the directory ``dataset/data``: 
    * ``test_features.p``
    * ``test_scene_labels.p``
    * ``training_features.p``
    * ``training_scene_labels.p``
    * ``validation_features.p``
    * ``validation_scene_labels.p`` 
    
* unzip the file ``AUDASC_pretrained_weights.zip``. This will produce the following files, which
will have to be place inside the directory ``pretrained_weights``:
    *  ``label_classifier.pytorch``
    * ``LICENSE``
    * ``non_adapted_cnn.pytorch`` 
    * ``target_cnn.pytorch``

That's it! 

You can either refer to the 
[ reproduce the results of the paper ](#reproduce-the-results-of-the-paper) section for 
reproducing the results presented in our paper, or to the
[ use the code with your own data ](#use-the-code-with-your-own-data) section if you want to
use our code for your own task and/or with your own data. 

Enjoy! 

---

## Reproduce the results of the paper

To reproduce the results of the paper, you have to:
* make sure that you have followed the steps in the 
[dependencies, pre-requisites, and setting up the code](#dependencies-pre-requisites-and-setting-up-the-code)
section
* be at the root directory of the project (i.e. in the ``undaw`` directory)
* issue the following command at your terminal: ``./undaw_paper.sh``

If you find any problem doing the above, please let us know through the 
[issues section](https://github.com/dr-costas/undaw/issues) of this
repository. 

---

## Use the code with your own data

To use your code with your own data, you will have to:

* provide a set of features to be used
* modify the ``data_handlers._domain_dataset.DomainDataset`` class
* modify the modules used and are in the ``modules`` package
* modify the settings to be used (i.e. the file that you will use and 
will be in the ``settings`` directory)
* modify the settings reading for each of the modules, by modifying the
functions in the ``helpers._models.py`` and ``helpers._modules_functions.py``
files 

To use the code with your new settings, you will have to place the new settings 
file in the ``settings`` directory and specify the new settings file at the
command line, when calling the ``main.py``. For example, like: 

    python scripts/main.py --config-file new_settings_file

Notice that the file name is without extension, meaning that only YAML
(i.e. `*.yaml` extension) files can be used.  

The processes (i.e. pre-training, adaptation, and evaluation) should be run 
with any module/neural network. 

If you have any question, please ask it using the 
[issues section](https://github.com/dr-costas/undaw/issues) of this
repository.

---

## Previous work

Our work is based on the following previous work: 
* AUDASC method, for domain adaptation for acoustic scene classification. 
* Wasserstein GANs, for using the WGAN algorithm.

--- 

## Acknowledgement

* Part of the computations leading to these results were performed on a TITAN-X GPU
donated by [NVIDIA](https://www.nvidia.com/en-us/) to K. Drossos. 
* The authors wish to acknowledge [CSC-IT Center for Science](https://www.csc.fi/), 
Finland, for computational resources. 
* The research leading to these results has received funding from the [European Research 
Council](https://erc.europa.eu/) under the European Union’s H2020 Framework Programme 
through ERC Grant Agreement 637422 EVERYSOUND. 
* P. Magron is supported by the [Academy of Finland](http://www.aka.fi/en), project no. 290190.
