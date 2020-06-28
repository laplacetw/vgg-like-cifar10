![](https://img.shields.io/github/license/laplacetw/vgg-like-cifar10)
# vgg-like-cifar10
Do deep learning classification on the [CIFAR-10 database](https://www.cs.toronto.edu/~kriz/cifar.html) with VGG-like structure approach 93% validation accuracy.

### Keras Code Sample
It's referenced the structure of [Keras code sample of CIFAR-10](https://keras.io/examples/cifar10_cnn/).<br>
<img src="./model_summary/keras_sample_cifar10.png" height="350px"><br>
Validation Accuracy : ≈78%<br>
<img src="./train_history/keras_sample_cifar10.png" height="350px">

### Advanced Solution
Next, let's referenced [the tutorial from Jason Brownlee PhD](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/), who is a professional developer and machine learning practitioner.<br>
<img src="./model_summary/jason_brownlee_cifar10.png" height="350px"><br>
Validation Accuracy : ≈89% (Train / Test : 94.60% / 89.35%)
<img src="./train_history/jason_brownlee_cifar10.png" height="350px">


### Factional Max-Pooling
Ref. [https://arxiv.org/abs/1412.6071](https://arxiv.org/abs/1412.6071)<br>
Due to VGG-16 or ResNet-50 are so giant and deep neural network, I tried fractional max-pooling after read the research paper to make a deeper VGG-like neural network but smaller and  shallower than VGG-16 and ResNet-50.To reduce overfitting, we use global average pooling layer instead of full connection layer.<br>
<img src="./model_summary/fmp_cifar10.png" height="350px"><br>
Validation Accuracy : ≈93% (Train / Test : 97.94% / 93.55%)
<img src="./train_history/fmp_cifar10.png" height="350px">

### Simple Comparison
|            | VGG-16/VGG-19 |VGG-like + FMP|
|:----------:|:-------------:|:------------:|
|total params|10M+           |1.5M          |
| model size |200+ MB        |12.4M         |