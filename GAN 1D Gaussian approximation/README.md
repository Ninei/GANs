# GANs
GANs(Generative Adversarial Networks) Exercise & Test

## Introduce
To get a better understanding of how this all works, we’ll use a GAN to solve a toy problem in TensorFlow – learning to approximate a 1-dimensional Gaussian distribution.

Reference Project : https://github.com/AYLIEN/gan-intro

Reference Blog : http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
    
## Training

To run without minibatch discrimination (and plot the resulting distributions):

    $ python ODG.py

To run with minibatch discrimination (and plot the resulting distributions):

    $ python ODG.py --minibatch