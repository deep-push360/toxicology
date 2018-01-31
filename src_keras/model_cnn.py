"""Contributors: Kayode Olaleye"""

from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from utils import load_keras_data

train_data_file = 'PATH_TO_CSV_FILE'

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_keras_data(train_data_file)

print('Shape of x: {}\n'.format(x.shape))
print('Shape of y: {}\n'.format(y.shape))
print('Size of vocabulary: {}\n'.format(len(vocabulary)))
print('Size of vocabulary_inv: {}\n'.format(len(vocabulary_inv)))

x_train, x_val, y_train, y_val = train_test_split( x, y, test_size=0.2, random_state=42)

print('Shape of x_train: {}\n'.format(x_train.shape))
print('Shape of y_train: {}\n'.format(y_train.shape))
print('Shape of x_val: {}\n'.format(x_val.shape))
print('Shape of y_val: {}\n'.format(y_val.shape))

sequence_length = x.shape[1] 
print('Sequence Length: {}'.format(x.shape[1]))
vocabulary_size = len(vocabulary_inv)
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 10
batch_size = 64
      
# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
print('RESHAPE', reshape)
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
print('conv_0', conv_0)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=7, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)
print("Summary of the model")
model.summary()
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(x_val, y_val))  # starts training