# Image Anomaly Detection using Autoencoders

This repository contains code for detecting anomalies in images using autoencoder-based anomaly detection. The model is trained on a dataset containing images and their corresponding sketches, and it aims to identify anomalous instances based on their reconstruction errors.

## Overview

Anomaly detection plays a crucial role in various applications such as fraud detection, defect identification, and outlier detection. In this project, we leverage autoencoder neural networks to detect anomalies in images. The autoencoder is trained on normal instances only and learns to reconstruct them accurately. During inference, instances with high reconstruction errors are flagged as anomalies.

## Dataset

The dataset consists of pairs of images and their corresponding sketches. Each pair represents a normal instance in the dataset, and anomalies are instances that deviate significantly from this norm. The dataset is divided into training, validation, and test sets to train and evaluate the anomaly detection model.

## Model Architecture

We use a convolutional autoencoder architecture for anomaly detection. The encoder network compresses the input images into a latent space representation, while the decoder network reconstructs the original images from this representation. The model is trained to minimize the reconstruction error, which is computed as the difference between the input and output images.

## Training

The autoencoder is trained using only normal instances from the dataset. During training, we aim to minimize the reconstruction loss on the validation set. We experiment with different architectures, hyperparameters, and training strategies to optimize the model's performance.

## Inference

During inference, new instances are fed through the trained autoencoder, and their reconstruction errors are calculated. Instances with reconstruction errors exceeding a predefined threshold are classified as anomalies. We evaluate the model's performance on a separate test set containing both normal and anomalous instances.

Clone the repository:

https://github.com/ashu3984/anomaly_detection_autoencoder.git
