# ML-implementations


The list of implementations and datasets.

1. "neural_network_from_scratch.py" builds from scratch a multilayer neural network for a multi-class classification problem. 
The script does not use any ready libraries and performs backpropagation for training. It uses "data/Image_Segmentaion.csv" file for training, the description of which can be found in "data/Image_Segmentaton_Description.pdf". The accuracies of trained neural network with various structures are depicted in "results/nn_from_scratch.png".   
![nn_from_scratch_results](https://user-images.githubusercontent.com/25514362/90460360-9a50e180-e0d1-11ea-9ae2-96e1e1c6d81d.png)

2. "autoencoder.py" creates autoencoder model to remove noise from images. It downloads MNIST dataset online and adds Gaussian noise to images for training purposes. The results of training can be found in "results/autonecoder0.1.png" and  "results/autoencoder0.5.png" files with noise factors 0.1 and 0.5 respectively. 
![autoencoder0 1](https://user-images.githubusercontent.com/25514362/90460221-742b4180-e0d1-11ea-870c-fdfef4bb4d52.png)
![autoencoder0 5](https://user-images.githubusercontent.com/25514362/90460225-74c3d800-e0d1-11ea-89a1-6f4389514886.png)

3. "resnet_fc.py" performs multi-class classification on Birds dataset using pretrained ResNet. After data augmentation, images are passed through pretrained Resnet with additional not-trained layers. The results of training can be found in "results/fc_loss.png". 
![fc_loss](https://user-images.githubusercontent.com/25514362/90460454-cec49d80-e0d1-11ea-914c-665222892073.png)
