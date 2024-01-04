[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/UwpqMYOQ)
# e4040-2023Fall-project
## Re-implementation and Analysis of SimCLR: A Contrastive Learning Framework

## Description
This repository contains a reimplementation of the SimCLR paper **A Simple Framework for Contrastive Learning of Visual Representations** by Chen et al., a framework for contrastive learning of visual representations. Our project specifically focuses on different ResNet models: ResNet50, ResNet101, and ResNet152. Each model directory contains its utilities, Jupyter Notebooks (`*.ipynb`) for model architecture, and saved models with various batch sizes.

## Structure
The repository is organized into the following directories:

- `resnet50/`: Contains utilities, notebooks, and models for the ResNet50 architecture.
- `resnet101/`: Contains utilities, notebooks, and models for the ResNet101 architecture.
- `resnet152/`: Contains utilities, notebooks, and models for the ResNet152 architecture.

Each directory has:
- `utils/`: Utility functions and scripts used across the models.
- `*.ipynb`: Jupyter Notebooks detailing the model architecture and training process.
- `models/` Repository of the models used, for `resnet101/` and `resnet152/` Accessible through : https://drive.google.com/drive/folders/1ygouFWmFboOuLGpt6UVCKgvp7JOH4gxK?usp=drive_link
- `requirements.txt`: Required packages and their version.

## Utils Folders:
There is a single `utils/` folder within every directory for different resnet models ( `resnet50/`, `resnet101/`, `resnet152/`). Each utils folder is composed of the following files with various adjustments to the inputs or parameters. 

### data_augment.py
This module focuses on data augmentation specifically designed for contrastive training in deep learning models. Key functionalities include:

- **gauss2D**: Generates a 2D Gaussian kernel. This is a foundational function used in Gaussian blur filtering.
- **gaussFilter**: Creates a Gaussian blur filter model. It utilizes the `gauss2D` function to apply Gaussian blur to images with customizable kernel size and sigma.
- **SimCLRDataGenerator**: A class providing advanced data augmentation techniques for contrastive learning. It includes several methods:
  - **Constructor**: Initializes the data generator with options for batch size and Gaussian blur parameters.
  - **random_apply**: Applies a given function to an input with a specified probability.
  - **color_distortion**: Distorts the color of the image in various ways (brightness, contrast, saturation, and hue).
  - **custom_augment**: Applies a series of augmentations including random cropping, resizing, flipping, color distortion, and Gaussian blur.
  - **generate**: Generates batches of augmented images, yielding two augmented versions of each image for contrastive learning.
  - **show_augmented_images**: A static method to display pairs of augmented images for visualization purposes.
  
 This file is integral for implementing data augmentation strategies mentioned in contrastive learning research, especially in the context of SimCLR architecture.

  ### encoder_projection.py

This module is responsible for building and handling the encoder and projection head of a SimCLR model, using TensorFlow and the Keras API. Key functionalities include:

- **SimCLREncoder Class**: A class to create and manage the SimCLR encoder and projection head.
  - **Constructor**: Initializes the encoder and projection head models. It takes the input shape and a flag indicating whether the base model (ResNet152) is trainable.
  - **create_encoder**: Constructs the encoder model using ResNet152 as a base and adds a global average pooling layer.
  - **create_projection_head**: Builds the projection head model, which is a simple feedforward neural network with dense layers.
  - **process_batch**: Processes a batch of data through both the encoder and the projection head, yielding the projected representations. This method is designed to work with a data generator.
  - **save_models**: Saves the encoder and projection head models to specified file paths.
  - **load_models**: Static method to load encoder and projection head models from given file paths.
  - **set_models**: Sets the encoder and projection head models with given model instances.

This module plays a crucial role in constructing the feature extraction and projection components of the SimCLR architecture, pivotal for self-supervised contrastive learning tasks.

### loss_function.py

This module contains the implementation of the loss function used in the SimCLR framework known as `nt_xent_loss`, specifically designed for contrastive learning in TensorFlow. The key functionalities are:

- **cosine_similarity**: 
  - **Purpose**: Computes the cosine similarity for pairs of examples in a given feature tensor.
  - **Inputs**: A tensor of shape (2N, d), where N is the number of examples and d is the feature dimension.
  - **Outputs**: A cosine similarity matrix of shape (2N, 2N).

- **nt_xent_loss** (Normalized Temperature-scaled Cross Entropy Loss):
  - **Purpose**: Implements the NT-Xent loss function, a core component in the SimCLR framework.
  - **Inputs**: A tensor of shape (2N, d), representing 2N augmented views from N examples, and a float representing the temperature parameter.
  - **Outputs**: A scalar tensor representing the NT-Xent loss.

The `cosine_similarity` function is a utility used within the `nt_xent_loss` function to compute the similarity between all pairs of examples in a batch. The NT-Xent loss is a contrastive loss function that encourages the model to learn similar representations for different augmented views of the same example while distinguishing between representations of different examples.

This loss function is central to the effectiveness of the SimCLR self-supervised learning approach.

## ResNet152
In addition to utils, this folder is composed of `Resnet152_Outputs.ipynb` notebook and an additional file under `utils\ResNet152run.py`. 
- **run_Resnet152**: 
  - **Purpose**: Orchestrates the entire training process using augmented data from CIFAR-10, the SimCLR encoder, and projection head. It also implements the training loop with NT-Xent loss and Adam optimizer.
  - **Parameters**: Includes train generator, learning rate, number of epochs, batch size, and batches limit.
  - **Outputs**: Trains the model and saves the encoder and projection head at specified paths.

- **extract_features**: 
  - **Purpose**: Extracts features from images using the provided encoder and projection head models.
  - **Inputs**: Encoder model, data generator, loaded encoder model, and loaded projection head.
  - **Outputs**: Returns features and labels extracted from the provided dataset.

- **get_accuracy**: 
  - **Purpose**: Evaluates the accuracy of the SimCLR model on the CIFAR-10 dataset.
  - **Inputs**: Paths to the saved encoder and projection head models.
  - **Functionality**: Loads the models, prepares the CIFAR-10 dataset, and extracts features. It then trains a supervised classification model on these features and evaluates its accuracy on the test set.

## Resnet101
Load the encoder and projection model based on below. 

- **extract_features**: 
  - **Purpose**: Extracts features from images using the provided encoder and projection head models.
  - **Inputs**: Encoder model, data generator, loaded encoder model, and loaded projection head.
  - **Outputs**: Returns features and labels extracted from the provided dataset.

Summary of results after running different Contrastive learning parameters.

![image](https://github.com/ecbme4040/e4040-2023Fall-Project-hehe-da3109-sdp2157-yx2771/assets/84935969/6c46ad63-a0c2-423d-acd8-d381514ced4d)

## ResNet50
Simplest model. Contains:
- **extract_features**: 
  - **Purpose**: Extracts features from images using the provided encoder and projection head models.
  - **Inputs**: Encoder model, data generator, loaded encoder model, and loaded projection head.
  - **Outputs**: Returns features and labels extracted from the provided dataset.



## Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/ecbme4040/e4040-2023Fall-Project-hehe-da3109-sdp2157-yx2771.git
cd gh repo clone ecbme4040/e4040-2023Fall-Project-hehe-da3109-sdp2157-yx2771
```

## Usage
To use the models or run the notebooks, navigate to the specific model directory and start the Jupyter Notebook:
```bash
cd resnet50  # Change this to resnet101 or resnet152 as needed
jupyter notebook
```
## Features 

- Detailed Jupyter Notebooks for understanding and implementing the SimCLR model architectures.
- Pre-trained models with different batch sizes for comparative analysis and further research.
- Extensive utility scripts for streamlined model training and evaluation.

## References

- Tensorflow CIFAR 10 Dataset. [https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data]
- Gradient [https://github.com/tensorflow/tensorflow/issues/31273]

# Organization of this directory
To be populated by students, as shown in previous assignments.
TODO: Create a directory/file tree


```
.
├── ECBM4040_team_hehe_FinalReport.pdf
├── Initial_model_training_on_Imagenet.ipynb
├── README.md
├── resnet101
│   ├── requirements.txt
│   ├── resnet101_32_1.ipynb
│   ├── resnet101_32_2.ipynb
│   ├── resnet101_32_3.ipynb
│   ├── resnet101_512_1.ipynb
│   ├── resnet101_512_2.ipynb
│   ├── resnet101model_32_1
│   │   ├── encoder_model
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       └── variables.index
│   │   ├── projection_head_model
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       └── variables.index
│   │   └── requirements.txt
│   ├── resnet101model_32_2
│   │   ├── encoder_model
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       └── variables.index
│   │   └── projection_head_model
│   │       ├── keras_metadata.pb
│   │       ├── saved_model.pb
│   │       └── variables
│   │           └── variables.index
│   ├── resnet101model_32_3
│   │   ├── encoder_model
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       └── variables.index
│   │   └── projection_head_model
│   │       ├── keras_metadata.pb
│   │       ├── saved_model.pb
│   │       └── variables
│   │           └── variables.index
│   ├── resnet101model_512
│   │   ├── encoder_model
│   │   │   ├── keras_metadata.pb
│   │   │   ├── saved_model.pb
│   │   │   └── variables
│   │   │       └── variables.index
│   │   └── projection_head_model
│   │       ├── keras_metadata.pb
│   │       ├── saved_model.pb
│   │       └── variables
│   │           └── variables.index
│   └── utils
│       ├── __pycache__
│       │   ├── data_augment.cpython-311.pyc
│       │   ├── data_augment.cpython-36.pyc
│       │   ├── data_augment.cpython-38.pyc
│       │   ├── encoder_projection.cpython-311.pyc
│       │   ├── encoder_projection.cpython-36.pyc
│       │   ├── encoder_projection.cpython-38.pyc
│       │   ├── loss_function.cpython-311.pyc
│       │   ├── loss_function.cpython-36.pyc
│       │   └── loss_function.cpython-38.pyc
│       ├── data_augment.py
│       ├── encoder_projection.py
│       └── loss_function.py
├── resnet152
│   ├── Resnet152_Outputs.ipynb
│   ├── requirements.txt
│   └── utils
│       ├── ResNet152run.py
│       ├── __pycache__
│       │   ├── ResNet152run.cpython-311.pyc
│       │   ├── data_augment.cpython-311.pyc
│       │   ├── data_augment.cpython-36.pyc
│       │   ├── data_augment2.cpython-311.pyc
│       │   ├── data_augment_cipar10.cpython-311.pyc
│       │   ├── encoder_projection.cpython-311.pyc
│       │   ├── encoder_projection.cpython-36.pyc
│       │   ├── loss_function.cpython-311.pyc
│       │   └── loss_function.cpython-36.pyc
│       ├── data_augment.py
│       ├── encoder_projection.py
│       └── loss_function.py
└── resnet50
    ├── requirements.txt
    ├── resnet50 model.ipynb
    ├── resnet50model
    │   ├── encoder_model
    │   │   ├── fingerprint.pb
    │   │   ├── keras_metadata.pb
    │   │   ├── saved_model.pb
    │   │   └── variables
    │   │       ├── variables.data-00000-of-00001
    │   │       └── variables.index
    │   └── projection_head_model
    │       ├── fingerprint.pb
    │       ├── keras_metadata.pb
    │       ├── saved_model.pb
    │       └── variables
    │           ├── variables.data-00000-of-00001
    │           └── variables.index
    └── utils
        ├── __pycache__
        │   ├── data_augment.cpython-311.pyc
        │   ├── data_augment.cpython-36.pyc
        │   ├── encoder_projection.cpython-311.pyc
        │   ├── encoder_projection.cpython-36.pyc
        │   ├── loss_function.cpython-311.pyc
        │   └── loss_function.cpython-36.pyc
        ├── data_augment.py
        ├── encoder_projection.py
        └── loss_function.py

35 directories, 82 files
```
