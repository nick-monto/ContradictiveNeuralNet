import os
import fnmatch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

img_height, img_width = 120, 200
num_epochs = 5
TRAINING_FOLDER = 'Input_spectrogram/Training/'
VALIDATION_FOLDER = 'Input_spectrogram/Validation/'

checkpoint = ModelCheckpoint('./weights.best.hdf5',
                             monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Define the model: 4 convolutional layers, 4 max pools
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(img_height, img_width, 1),
                 strides=(2, 2),
                 activation='relu',
                 kernel_initializer='TruncatedNormal',
                 name='conv1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

model.add(Conv2D(64, (3, 3), padding='same',
                 strides=(2, 2),
                 activation='relu',
                 kernel_initializer='TruncatedNormal',
                 name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu',
                 kernel_initializer='TruncatedNormal',
                 name='conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))

model.add(Conv2D(256, (3, 3), padding='same',
                 activation='relu',
                 kernel_initializer='TruncatedNormal',
                 name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))

model.add(Flatten(name='flat1'))  # converts 3D feature mapes to 1D feature vectors
model.add(Dense(256, activation='relu',
                kernel_initializer='TruncatedNormal', name='fc1'))
model.add(Dropout(0.5, name='do1'))  # reset half of the weights to zero

model.add(Dense(70, activation='softmax', name='output'))

model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])


#Load in the images and fit the model
def load_image(path):
    img = Image.open(path).convert('L')  # read in as grayscale
    img = img.resize((img_width, img_height))
    # img.load()  # loads the image into memory
    img_data = np.asarray(img, dtype="float")
    img_data = img_data*(1./255)
    return img_data

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result[0]


df_train = pd.read_table('img_set_train.txt',
                         delim_whitespace=True,
                         names=['stimulus', 'word'])

df_val = pd.read_table('img_set_val.txt',
                       delim_whitespace=True,
                       names=['stimulus', 'word'])

stim_val = df_val['stimulus']
labels_val = pd.get_dummies(df_val['word'])
labels_val = labels_val.values

# Load up all the validation files, the training files will be batch loaded and trained
specs_val_input = []
for i in range(len(stim_val)):
    specs_val_input.append(load_image(find(stim_val.iloc[i],
                                                                        VALIDATION_FOLDER)))
specs_val_input = np.asarray(specs_val_input)
specs_val_input = specs_val_input.reshape((len(stim_val),
                                                                        img_height, img_width, 1))
print('There is a total of {} training stimuli!'.format(len(df_train)))
print('There is a total of {} validation stimuli!'.format(specs_val_input.shape[0]))

shuffled_train = df_train.sample(frac=1, random_state=42)


count=0
while count < 35000:
    train_slice = shuffled_train.iloc[count:count+5000]
    stim_train = train_slice['stimulus']
    labels_train = pd.get_dummies(train_slice['word'])
    labels_train = labels_train.values
    specs_train_input = []
    print('Loading set of training data...')
    for i in range(len(stim_train)):
        specs_train_input.append(load_image(find(stim_train.iloc[i],
                                                 TRAINING_FOLDER)))
    specs_train_input = np.asarray(specs_train_input)
    specs_train_input = specs_train_input.reshape((len(stim_train),
                                                   img_height, img_width, 1))
    print('Successfully loaded set of {} training spectrograms!'.format(len(stim_train)))
    # Model fitting
    model.fit(x=specs_train_input, y=labels_train,
                    batch_size=32, epochs=2,
                    verbose=1, callbacks=callbacks_list,
                    validation_split=0., validation_data=(specs_val_input, labels_val),
                    shuffle=False, class_weight=None, sample_weight=None,
                    initial_epoch=0, steps_per_epoch=None, validation_steps=None)

    count = count + 5000

model.save('ContradictiveNet_4Conv.h5')
