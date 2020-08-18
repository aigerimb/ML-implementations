# ML-implementations


The list of implementations and datasets.

1. "neural_network_from_scratch.py" builds from scratch a multilayer neural network for a multi-class classification problem. 
The script does not use any ready libraries and performs backpropagation for training. It uses "data/Image_Segmentaion.csv" file for training, the description of which can be found in "data/Image_Segmentaton_Description.pdf". The accuracies of trained neural network with various structures are depicted in "results/nn_from_scratch.png".   
2. "autoencoder.py" creates autoencoder model to remove noise from images. It uses MNIST dataset and adds Gaussian noise to images for training purposes. The results of training can be found in "results/autonecoder0.1.png" and  "results/autoencoder0.5.png" files with noise factors 0.1 and 0.5 respectively. 
