Problem Statement:

In real-world scenarios, maintaining awareness of the individuals in our vicinity is essential.     Face detection plays a vital role in gathering information about individuals present at a specific location and time, particularly for real-time identification of human faces captured by cameras. Furthermore, we aim to analyze the facial expressions of individuals within the frame to gain deeper insights into their emotions and reactions.


Data Collection:

https://www.kaggle.com/datasets/msambare/fer2013

Preprocessing:


The dataset comprises images of uniform size (48x48 pixels) and grayscale format. 
It is structured into directories for both training and validation, further categorized into seven distinct classes.
 Each class contains a sufficient number of images for training purposes, although the quantity may differ across classes. 
Therefore, data augmentation techniques can be applied to certain classes to enhance training diversity. 
The dataset's organization is well-structured, eliminating the need for any preprocessing steps


Algorithm

The Basic task is to detect the faces:


1)Algorithm: Haar Cascade Classifiers - Pre trained

Using the Haar Cascade Classifier to detect the faces present in front of the camera and surrounding the detected faces with a rectangular box.

2)Trained CNN - Emotion Detection

Training our own CNN Model for predicting the emotions of the individuals. DeepFace is used for age and gender prediction on the faces determined using Haar Cascade Classifiers


CNN Model- 
Contains 6 Convolutional layers with kernel size (3*3) with varying filters and 5 MaxPooling layers with kernel size (2*2). Finally, two dense layer and a softmax layer – for predicting the output.



Training- 
The model is trained for 50 epochs with a batch size of
64.	Image size – 48*48. Images are Gray scaled.


Performance

Metrics – Accuracy, Categorical Cross Entropy Loss.


•	Training Accuracy – 65.82%
•	Training Loss – 1.5118
•	Validation Accuracy – 65.72%
•	Validation Loss – 1.5352


