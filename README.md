# Stochastic differential equations lab
This repository contains code for the course SF2522.

## Requirements
Make sure to have installed:

* Python 3.6 or later
* pip 3

## Installation
Install the requirements
```sh
$ pip install -r requirements.txt
```

## Homework 4
To run the code use
```sh
$ python tf_lab.py
```
To change the number of dimensions edit the variable `d` on line 6.

## Homework 5
Tasks 1, 2, 3, 4 and 5 are solved with the code in `tf_mnist_lab.py`. To run please use

```sh
$ python tf_mnist_lab.py
```

Task 6 is implemented using a neural network and using a convolutional neural network. To run the neural network please
use

```sh
$ python improved_nn.py
```

To run the convolutional neural network please use (please note that this takes up ~30 minutes to run on CPU, if 
possible use GPU version of TensorFlow)

```sh
$ python cnn.py
```
 

