# ContradictiveNeuralNet
Studying time-course activation patterns in isolated word recognition using a 2D convolutional neural net. Talk about unconventional... 

## Process
 1. Start with a bare bones CNN structure that performs word recognition well using my standard input creation pipeline
    * Setting the X-axis of the spectrograms to be static: length of longest word
 2. Modify the filter size of the CNN structure to encompass the full height of the image
    * this will 'mimic' time-course of input
