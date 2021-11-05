# Kinship-Verification
Kinship Verification Through Facial Images Using CNN-Based Features 

In this project I implement an algorithm which receive a pair of images and determine whether they have kinship relationship or not. this algorithm use one of the Svm or neural network approch for classify datasets. the algorithm consist of below steps :
1- Reading data
2- Feature extraction using Vgg-face model
3- Merging features
4- Fearure normalization
5- Feature selection using Fisher-score 
6- Classification using SVM or neural network

this project contains 3 directories:
1-models
2-pretrained
3-utils

Utils directory have reader.py which we can use for generating input file for algorithm. This directory also have preprcessor file which have some methods for preprocessing the data.

Pretrained directory have vgg_face file which consist of vgg-face model architecture.

Finally models contains two file, nn_model for the neural network implementation and svm_model for SVM implementation.