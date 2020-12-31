# Signed-MNIST-recognition
I developed a convolutional neural network model for recognising letters from the  [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist/ "Named link title"), consisting of 24 American Sign Language (ASL) letters (excluding those that require motion). This CNN uses 4 convolutional layers, 4 pooling layers, and 2 fully connected layers to accurately classify the input images in ASL as the relevant letter. The model achieves 99.986% accuracy within 25 epochs, with consistent accuracy > 99% within fifteen epochs, with runtime below ten minutes. 

The model was built in Kaggle for easy dataset import and operations, and the original script may be viewed [here](https://www.kaggle.com/marcusdeans/signed-mnist-cnn/ "Kaggle link"), along with the execution output. 

Libraries implemented within this project:
* Keras
* TensorFlow
* Numpy
* Pandas
