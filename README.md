# ContradictiveNeuralNet
Studying time-course activation patterns in isolated word recognition using a 2D convolutional neural net. Talk about unconventional...

## Process
 1. Create training and test set from a random 50 items.
    * For training, each item will have 350 iterations of creation with white noise
    * Test will consist of the original items from training
 2. Start with a bare bones CNN structure that performs word recognition well
    * Setting the X-axis of the spectrograms to be static: length of longest word; though probably not because then the network might rely on the added silence.
 2. Modify the filter size of the CNN structure to encompass the full height of the image
    * this will 'mimic' 'human processing' of time-dependent signal
