# Tied Weights Training in Multi-task Learning
Tied Weights training approach in the multi-task learning setting, presented in: 

Morfi, V. and Stowell, D. (2018). Deep learning for audio event detection and tagging on low-resource
datasets. Applied Sciences, 8(8):1397

## Brief Description
Tied weights training used for bird song detection in time and tagging of active bird species in a recording. The two tasks are named WHEN and WHO, respectively.

Tied weights training follows the hard parameter training convention, where layers and their weights are shared between tasks. However, in contrast to joint training, different types of input can be used to train each task. In this example of tied weights training, there is a Shared Convolutional Part that refers to the common convolutional and max pooling layers of WHEN and WHO tasks, and the weights between the two tasks are constrained to be identical in these layers. 

During training each network is trained consecutively for one epoch, updating the weights of the shared layers.

This example is specific to the NIPS4Bplus dataset and annotations. And the code assumes that the recordings have been pre-processed into melspecs of size (mel bands, time frames)=(40, 432)

![tied](https://user-images.githubusercontent.com/18617080/60804009-6ab47380-a174-11e9-9145-81803c29f844.png)

## NIPS4Bplus Dataset
[NIPS4Bplus recordings](http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz)
[NIPS4Bplus temporal annotations](https://figshare.com/articles/Transcriptions_of_NIPS4B_2013_Bird_Challenge_Training_Dataset/6798548)

