#################################
### Author: Veronica Morfi 
### Affiliation: C4DM, Queen Mary University of London
###
### Version date: July 8th 
###
### First presented in: Morfi, V. and Stowell, D. (2018). Deep learning for audio event detection and tagging on low-resource datasets. Applied Sciences, 8(8):1397. 
### Dataset: NIPS4Bplus https://figshare.com/articles/Transcriptions_of_NIPS4B_2013_Bird_Challenge_Training_Dataset/6798548
#################################

import librosa
import numpy as np
import pandas
import pickle
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, Callback
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import math

##############################
### data and target labels ###
##############################

# load data in melspec form
f = open('melspecs.pckl', 'r') # data_shape = (no_recs, mel_bins, time_frames)
MS = pickle.load(f)
f.close()

# load who labels; contain info about the species present in a recording
fwe = open('labels_who.pckl', 'r') # who_labels_shape = (no_recs, no_species)
weaklabels = pickle.load(fwe)
f.close()

# load when labels; recordings with any bird have 1 in all time frames, recordings with no birds have 0 in all time frames
fst = open('labels_strong.pckl', 'r') # when_labels_shape = (no_recs, time_frames)
stronglabels = pickle.load(fst)
f.close()

###########################################
### split to train/val and testing sets ###
###########################################

# train and validation sets: recordings start-498 including recordings from 499-end without full temporal annotations (total of 513 recordings)
# train and validation sets shape: (513, mel_bins, time_frames)
MS_train = MS[:499]
MS_train = np.concatenate((MS_train, MS[501:502], MS[514:515], MS[540:541], MS[544:545], MS[552:553], MS[567:568], MS[575:576], MS[587:588], MS[591:592], MS[599:600], MS[614:615], MS[649:651], MS[665:666]), axis=0)
weaklabels_train = weaklabels[:499]
weaklabels_train = np.concatenate((weaklabels_train, weaklabels[501:502], weaklabels[514:515], weaklabels[540:541], weaklabels[544:545], weaklabels[552:553], weaklabels[567:568], weaklabels[575:576], weaklabels[587:588], weaklabels[591:592], weaklabels[599:600], weaklabels[614:615], weaklabels[649:651], weaklabels[665:666]), axis=0)
stronglabels_train = stronglabels[:499]
stronglabels_train = np.concatenate((stronglabels_train, stronglabels[501:502], stronglabels[514:515], stronglabels[540:541], stronglabels[544:545], stronglabels[552:553], stronglabels[567:568], stronglabels[575:576], stronglabels[587:588], stronglabels[591:592], stronglabels[599:600], stronglabels[614:615], stronglabels[649:651], stronglabels[665:666]), axis=0)

# test set: recordings 499-end excluding ones without full temporal annotations (total of 174 recordings)
# test set shape: (174, mel_bins, time_frames)
MS_test = MS[499:501]
MS_test = np.concatenate((MS_test, MS[502:514], MS[515:540], MS[541:544], MS[545:552], MS[553:567], MS[568:575], MS[576:587], MS[588:591], MS[592:599], MS[600:614], MS[615:649], MS[651:665], MS[666:]), axis=0)
weaklabels_test = weaklabels[499:501]
weaklabels_test = np.concatenate((weaklabels_test, weaklabels[502:514], weaklabels[515:540], weaklabels[541:544], weaklabels[545:552], weaklabels[553:567], weaklabels[568:575], weaklabels[576:587], weaklabels[588:591], weaklabels[592:599], weaklabels[600:614], weaklabels[615:649], weaklabels[651:665], weaklabels[666:]), axis=0)
stronglabels_test = stronglabels[499:501]
stronglabels_test = np.concatenate((stronglabels_test, stronglabels[502:514], stronglabels[515:540], stronglabels[541:544], stronglabels[545:552], stronglabels[553:567], stronglabels[568:575], stronglabels[576:587], stronglabels[588:591], stronglabels[592:599], stronglabels[600:614], stronglabels[615:649], stronglabels[651:665], stronglabels[666:]), axis=0)

#####################
### Normalization ###
#####################

# train data (total of 450 recordings)
temp = np.reshape(MS_train[0:450], (-1,40)).T

meanT = np.mean(temp, axis=1)
meanT = meanT.reshape((-1,1))
stdT = np.std(temp, axis=1)
stdT = stdT.reshape((-1,1))

MS_train_norm = (temp-meanT)/stdT
MS_train_norm = MS_train_norm.T
MS_train_norm = np.reshape(MS_train_norm, MS_train[0:450].shape)

# validation data (total of 63 recordings)
MS_val = MS_train[450:]
tempv = np.reshape(MS_val, (-1, 40)).T

MS_val_norm = (tempv-meanT)/stdT
MS_val_norm = MS_val_norm.T
MS_val_norm = np.reshape(MS_val_norm, MS_val.shape)

# test data
tempt = np.reshape(MS_test,(-1,40)).T
MS_test_norm = (tempt-meanT)/stdT
MS_test_norm = MS_test_norm.T
MS_test_norm = np.reshape(MS_test_norm, MS_test.shape)


##########################
### network input prep ###
##########################
# reshape
TEMPt = np.empty((MS_train_norm.shape[0], MS_train_norm.shape[2], MS_train_norm.shape[1]))
for i in range(len(MS_train_norm)):
    TEMPt[i] = MS_train_norm[i].T
MS_train_norm = TEMPt 
MS_train_norm = np.expand_dims(MS_train_norm, axis=3)

TEMPv = np.empty((MS_val_norm.shape[0], MS_val_norm.shape[2], MS_val_norm.shape[1]))
for i in range(len(MS_val_norm)):
    TEMPv[i] = MS_val_norm[i].T
MS_val_norm = TEMPv
MS_val_norm = np.expand_dims(MS_val_norm, axis=3)

TEMPte = np.empty((MS_test_norm.shape[0], MS_test_norm.shape[2], MS_test_norm.shape[1]))
for i in range(len(MS_test_norm)):
    TEMPte[i] = MS_test_norm[i].T
MS_test_norm = TEMPte
MS_test_norm = np.expand_dims(MS_test_norm, axis=3)

# split train and validation labels
stronglabels_train = stronglabels_train[0:450]
weaklabels_train = weaklabels_train[0:450]

stronglabels_val = stronglabels_train[450:]
weaklabels_val = weaklabels_train[450:]

#####################################################
### positive-negative separation for WHEN network ###
#####################################################
# Separate positive and negative recordings and their labels (only for train set)
strongpos_labels = [] # all ones (temporal)
strongneg_labels = [] # all zeros (temporal)
weakpos_labels = [] # at least one one (tags)
weakneg_labels = [] # all zeros (tags)
pos_MS_train_norm = []
neg_MS_train_norm = []
for i in range(len(stronglabels_train)):
    if stronglabels_train[i][0] == 0:
        weakneg_labels.append(weaklabels_train[i])
        strongneg_labels.append(stronglabels_train[i])
        neg_MS_train_norm.append(MS_train_norm[i])
    else:
        weakpos_labels.append(weaklabels_train[i])
        strongpos_labels.append(stronglabels_train[i])
        pos_MS_train_norm.append(MS_train_norm[i])

strongpos_labels = np.asarray(strongpos_labels)
strongneg_labels = np.asarray(strongneg_labels)
weakpos_labels = np.asarray(weakpos_labels)
weakneg_labels = np.asarray(weakneg_labels)
pos_MS_train_norm = np.asarray(pos_MS_train_norm) # totals to 385 recs
neg_MS_train_norm = np.asarray(neg_MS_train_norm) # totals to 65 recs

############################
### Network architecture ###
############################
def convBNpr(a, num_filters, kernel):
    c1 = Conv2D(filters=num_filters, kernel_size=kernel, strides=(1, 1), padding='same', use_bias=False, kernel_initializer=glorot_uniform(seed=123),kernel_regularizer=regularizers.l2(0.001))(a)
    c1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(c1)
    c1 = Activation('relu')(c1)
    return c1

def A(input_shape):
    a = Input(shape=(input_shape)) 
    
    # CNN
    c1 = convBNpr(a, 64, (3,3))
    c1 = convBNpr(c1, 64, (3,3))
    p1 = MaxPooling2D(pool_size=(1, 5))(c1)
    
    c2 = convBNpr(p1, 64, (3,3))
    c2 = convBNpr(c2, 64, (3,3))
    p2 = MaxPooling2D(pool_size=(1, 4))(c2)
    
    c3 = convBNpr(p2, 64, (3,3))
    c3 = convBNpr(c3, 64, (3,3))
    p3 = MaxPooling2D(pool_size=(1, 2))(c3)
    
    model = Model(inputs=a, outputs=[p3])
    return model

def B(input_shape):
    p3 = Input(shape=(input_shape)) 

    # Reshape
    b = Reshape((input_shape[0],-1))(p3)
    
    # GRU 
    r1 = Bidirectional(GRU(units=64, kernel_regularizer=regularizers.l2(0.01), return_sequences=True))(b)
    r2 = Bidirectional(GRU(units=64, kernel_regularizer=regularizers.l2(0.01), return_sequences=True))(r1)
    
    # Dense 
    d1 = TimeDistributed(Dense(87, activation='relu', kernel_regularizer=regularizers.l2(0.01)))(r2)
    
    # WHEN labels
    d2 = TimeDistributed(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))(d1)
    f1 = Flatten()(d2)
    
    model = Model(inputs=p3, outputs=[f1])
    return model

def C(input_shape, species):
    p3 = Input(shape=(input_shape)) 
    
    a1 = GlobalAveragePooling2D()(p3)

    # WHO labels
    d5 = Dense(species, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), bias_initializer=random_normal(mean=-4.60, stddev=0.05, seed=123))(a1) # bias initialiser specific to our train data
        
    model = Model(inputs=p3, outputs=[d5])
    return model

##########################
### WHEN Loss function ###
##########################

def WHENLoss(yTrue,yPred): 
    # MMM loss function
    # a:mean=0.5, b:max=1 and c:min=0
    a = K.binary_crossentropy(tf.scalar_mul(0.5,K.max(yTrue, axis=-1)),K.mean(yPred, axis=-1))
    b = K.binary_crossentropy(K.max(yTrue, axis=-1), K.max(yPred, axis=-1))
    c = K.binary_crossentropy(K.min(tf.scalar_mul(0.0, yTrue), axis=-1), K.min(yPred, axis=-1))
    l_when = tf.scalar_mul(0.33,(a+b+c))
    return l_when

#####################
### Learning rate ###
#####################
def step_decay(epoch):
    initial_lrate = 1e-5 # 1e-3, 1e-4
    drop = 0.5
    epochs_drop = 20.0
    min_lrate = 1e-8
    lrate = np.maximum(initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop)),min_lrate)
    return lrate

class MyLearningRateScheduler(Callback):

    def __init__(self, schedule, verbose=0, ep=0):
        super(MyLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.ep = ep

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(self.ep)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (self.ep + 1, lr))
        self.ep += 1

#############
### Model ###
#############

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"]="0" # which GPU to use
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3 # what % of the GPU to use
set_session(tf.Session(config=config))

# define and compile models
adam1 = Adam(lr = 1e-5)
adam2 = Adam(lr = 1e-5)

model_A = A((432, 40, 1))
model_A.summary()

model_B = B((432, 1, 64)) # input of B is output of A
model_B.summary()

model_C = C((432, 1, 64), 87) # input of C is output of A
model_C.summary()

mel_input = Input(shape=((432, 40, 1)))

a_out_bc_input = model_A([mel_input])
b_out = model_B([a_out_bc_input])
c_out = model_C([a_out_bc_input])

model_when = Model([mel_input], b_out) # melspec input --> A --> B --> WHEN prediction
model_who = Model([mel_input], c_out) # melspec input --> A --> C --> WHO prediction

model_when.summary()
model_who.summary()

model_when.compile(loss=[WHENLoss], optimizer=adam1)
model_who.compile(loss=['binary_crossentropy'], optimizer=adam2)

#################
### Callbacks ###
#################

reduce_lr_who = MyLearningRateScheduler(step_decay, verbose=1)
reduce_lr_when = MyLearningRateScheduler(step_decay, verbose=1)

tbCallback_who = TensorBoard(log_dir='../logs/TB_who', write_graph=True)
tbCallback_when = TensorBoard(log_dir='../logs/TB_when', write_graph=True)

# saving weights every 10 epochs, because there is no way to properly early stop for WHEN network
cpCallback_who = ModelCheckpoint('./checkpoints/who/weights_who_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min', period=10)
cpCallback_when = ModelCheckpoint('./checkpoints/when/weights_when_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True, mode='min', period=10)

###############################
### WHEN training generator ###
###############################

def train_generator(pos_MS_train_norm, neg_MS_train_norm, pos_labels, neg_labels, batchsize=4): # actual batchsize is 4*2=8
    while 1:
        indexes = np.arange(len(pos_MS_train_norm))
        np.random.shuffle(indexes)
        
        imax = int(len(indexes)/batchsize)
        
        for i in range(imax):
            pos_out_data = []
            pos_out_labels =[]
            neg_out_data = []
            neg_out_labels =[]
            
            for k in indexes[i*batchsize:(i+1)*batchsize]:
                pos_out_data.append(pos_MS_train_norm[k])
                pos_out_labels.append(pos_labels[k])
            
            neg_ind = np.arange(len(neg_MS_train_norm))
            np.random.shuffle(neg_ind)
            
            for n in range(batchsize):
                ind = neg_ind[n]
                neg_out_data.append(neg_MS_train_norm[ind])
                neg_out_labels.append(neg_labels[ind])
            
            out_data = []
            out_labels = []
            
            for m in range(batchsize):
                out_data.append(pos_out_data[m])
                out_data.append(neg_out_data[m])
                out_labels.append(pos_labels[m])
                out_labels.append(neg_labels[m])
            
            yield np.asarray(out_data), np.asarray(out_labels)

################
### Training ###
################

for epoch in range(1500):

    model_when.fit_generator(train_generator(pos_MS_train_norm, neg_MS_train_norm, strongpos_labels, strongneg_labels), 
                    steps_per_epoch=len(pos_MS_train_norm)/4, epochs=epoch+1, verbose=1, 
                    callbacks=[tbCallback_when, reduce_lr_when, cpCallback_when], validation_data=(MS_val_norm, stronglabels_val),
                       initial_epoch=epoch)

    model_who.fit(x=MS_train_norm, y=weaklabels_train, batch_size=8, epochs=epoch+1, 
         callbacks=[tbCallback_who, reduce_lr_who, cpCallback_who], validation_split=0,
          validation_data=(MS_val_norm, weaklabels_val), shuffle=True, verbose=1,
              initial_epoch=epoch)

###############
### Testing ###
###############

path2weights_when = './checkpoints/when/'
saved_weights_when = [f for f in listdir(path2weights_when) if isfile(join(path2weights_when, f))]

path2weights_who = './checkpoints/who/'
saved_weights_who = [f for f in listdir(path2weights_who) if isfile(join(path2weights_who, f))]

# predictions from every 10 epochs
t_predictions_when = []
t_predictions_who = []
for w in saved_weights_when:
    # load weights of trained network
    model_when.load_weights(path2weights_when+w)
    # predict on testing set
    t_predictions_when.append(model_when.predict(x=MS_test_norm, batch_size=1, verbose=1))

for w in saved_weights_who:
    # load weights of trained network
    model_who.load_weights(path2weights_who+w)
    # predict on testing set
    t_predictions_who.append(model_who.predict(x=MS_test_norm, batch_size=1, verbose=1))

fw = open('./predictions/when.pckl', 'w')
pickle.dump(t_predictions_when, fw)
fw.close()

fw = open('./predictions/who.pckl', 'w')
pickle.dump(t_predictions_who, fw)
fw.close()
