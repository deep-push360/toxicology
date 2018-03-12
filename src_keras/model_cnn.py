"""Contributors: Kayode Olaleye
Implementation of the convolutional neural net. """

import matplotlib
matplotlib.use("Pdf")
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Embedding
#from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, Reshape, Concatenate, Dropout, GlobalMaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Dropout
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import MODELS_DIR, TENSORBOARD_DIR, OUTPUT_DIR
from utils import save_makedirs, save_model
print("keras: ", keras.__version__)
print("tensorflow: " ,tf.__version__)
np.random.seed(123)


def train_model(model, comments_train, labels_train, comments_val, labels_val, model_id, 
                out_path, nb_epoch=1, batch_size=32, checkpoints=False, 
                tensorboard=False, earlystop=False):
    """Train the model with the given comments and labels"""
    x_train, y_train, x_val, y_val =comments_train, labels_train, comments_val, labels_val
    
    #x_train = normalise_input(x_train)
    print('Shape of x_train: {}, Shape of y_train: {}'.format(x_train.shape, y_train.shape))
    # Directory which is used to store the model and its weights.
    model_dir = os.path.join(MODELS_DIR, model_id)
    
    earlystopper = None
    if earlystop:
        earlystop_file = os.path.join(model_dir, "early_weights.hdf5")
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    
    checkpointer = None
    if checkpoints:
        checkpoints_file = os.path.join(model_dir, "weights.hdf5")
        checkpointer = ModelCheckpoint(checkpoints_file)
        
    tensorboarder = None
    if tensorboard:
        log_dir = os.path.join(TENSORBOARD_DIR, model_id)
        tensorboarder = TensorBoard(log_dir=log_dir)
        
    callbacks = [c for c in [earlystopper, checkpointer, tensorboarder] if c]
    
    print("Start training.")
    
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks, 
                        validation_data=(comments_val, labels_val))

    plot_history(history, out_path)

    save_model(model, model_dir)
    return model 

def plot_history(history, out_path):
    print(history.history.keys())
    plot_accuracy(history, out_path)
    plot_loss(history, out_path)
    out_file = os.path.join(out_path, "history.pickle")
    with open(out_file, "wb") as out:
        pickle.dump({
                "acc": history.history['acc'],
                "val_acc": history.history['val_acc'],
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
                }, out)
        
#  "Accuracy"
def plot_accuracy(history, out_path):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(out_path, 'history_acc.png'))
    
# "Loss"
def plot_loss(history, out_path):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 0.9999)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(out_path, 'history_loss.png'))

def init_model(model_id, comments, embedding_matrix, vocab_size=1000, 
               architecture='one_layer', hidden_dims=200,
               embedding_dims=256, filters=250, filter_size=3, sequence_length=400,
               filter_size_1=3, filter_size_2=4, filter_size_3=5,
               num_filter_1=512, num_filter_2=512, num_filter_3=512,
               pool_stride_1=1,pool_stride_2=1, pool_stride_3=1,
               learning_rate=0.05, keepprob=0.5,
               momentum=0.9, decay=0.0):
    """ Initialise a new model with the given hyperparameters and save it for later use. """
    emb_init_weight = embedding_matrix
    inputs = Input(shape=(sequence_length,), dtype='int32')
    if architecture == 'base_arch':
#         #-----------------------------------------------------------------
     
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dims,
                              input_length=sequence_length)(inputs)
        reshape = Reshape((sequence_length,embedding_dims,1))(embedding)
        drop = Dropout(keepprob)(reshape)
        conv_1 = Conv2D(filters, kernel_size=(filter_size, embedding_dims), 
                        padding='valid', kernel_initializer='normal', activation='relu')(drop)
        maxpool = GlobalMaxPooling2D()(conv_1)
        dense_1 = Dense(units=hidden_dims, activation='relu')(maxpool)
        dropout = Dropout(keepprob)(dense_1)
        output = Dense(units=6, activation='softmax')(dropout)
        
    if architecture == 'base_glove_arch':
#         #-----------------------------------------------------------------
     
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dims,
                              input_length=sequence_length, 
                              embeddings_initializer=[emb_init_weight])(inputs)
        reshape = Reshape((sequence_length,embedding_dims,1))(embedding)
        drop = Dropout(keepprob)(reshape)
        conv_1 = Conv2D(filters, kernel_size=(filter_size, embedding_dims), 
                        padding='valid', kernel_initializer='normal', activation='relu')(drop)
        maxpool = GlobalMaxPooling2D()(conv_1)
        dense_1 = Dense(units=hidden_dims, activation='relu')(maxpool)
        dropout = Dropout(keepprob)(dense_1)
        output = Dense(units=6, activation='softmax')(dropout)
        
            
    elif architecture == 'deep_arch':
        #-----------------------------------------------------------------
        embedding = Embedding(input_dim=vocab_size, 
                              output_dim=embedding_dims, input_length=sequence_length)(inputs)
        reshape = Reshape((sequence_length,embedding_dims,1))(embedding)
        drop = Dropout(keepprob)(reshape)
        conv_1 = Conv2D(num_filter_1, kernel_size=(filter_size_1, embedding_dims), 
                        padding='valid', kernel_initializer='normal', activation='relu')(drop)
        conv_2 = Conv2D(num_filter_2, kernel_size=(filter_size_2, embedding_dims), 
                        padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_3 = Conv2D(num_filter_3, kernel_size=(filter_size_3, embedding_dims), 
                        padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_1 = GlobalMaxPooling2D()(conv_1)
        maxpool_2 = GlobalMaxPooling2D()(conv_2)
        maxpool_3 = GlobalMaxPooling2D()(conv_3)

        concatenated_tensor = Concatenate(axis=1)([maxpool_1, maxpool_2, maxpool_3])
        #flatten = Flatten()(maxpool)
        dense_1 = Dense(units=hidden_dims, activation='relu')(concatenated_tensor)
        dropout = Dropout(keepprob)(dense_1)
        output = Dense(units=6, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    model = compile_model(model, learning_rate, momentum, decay)
    
    # Print a summary of the model to the console.
    print("Summary of the model")
    model.summary()
    
    model_dir = os.path.join(MODELS_DIR, model_id)
    save_makedirs(model_dir)
    
    save_model(model, model_dir)
    
    return model

def compile_model(model, learning_rate, momentum, decay):
    """ Compile the keras model with the given hyperparameters."""
    #optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay)
    #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model