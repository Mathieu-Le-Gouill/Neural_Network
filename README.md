# Neural_Network
Here is some code of a C++ neural network.


  *What is an Artificial Neural Network ?

    -Artificial Neural Network is used in a particuliar domain of the machine learning : the deep learning.
     It's concept is actually inspired by the human brain, by simply replacing the synapses and neurons by numbers.

    -The network is composed of layers, themselves composed of neurons.
     The layers are linked by what we called weights. A given neuron in a layer is linked with N numbers weights where N is the number of neurons in the previous layer.
     So that each neuron in a given layer affect all the next layers neurons depending of the weights value.
     Here is a link to get a better understanding of the nn : http://neuralnetworksanddeeplearning.com/chap1.html

  *How works a Neural Network ?

    -By processing data and the targeted output, the Neural Network use stochastic gradient descent to reduce how wrong the nn's output is between its current result and the            targeted value. So that it can reduce the error it got at each new steps.

    -The nn work in two different ways :
        Forward Propagation : Where it compute the network output by processing the given input in each layers each one depending of the former.
        Backward Propagation : Where it compute the loss gradient of each layers so that it can use the gradient descent latter on.
