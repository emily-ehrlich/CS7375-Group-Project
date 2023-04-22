import pandas as pd
import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
import librosa.display

from keras import layers
from keras import models
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# variable to hold number of mfcc samples to take
n_mfcc = 40

# Set the path to the full UrbanSound dataset
audio_files = './UrbanSound8K/audio/'
# Set path to metadata csv file included with UrbanSound dataset
metadata = pd.read_csv('./UrbanSound8k/metadata/UrbanSound8K.csv')


# function to generate mfcc values from audio files
def extract_features(file_name):
    # librosa.load() automatically normalizes audio, converts to mono, and sets sample rate to 22050
    audio, sample_rate = librosa.load(file_name)
    # compute mfccs from audio files. Returns a 2D array where each column corresponds to a time frame and each row
    # is a mfcc
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    # transpose mfccs array so that rows and columns are swapped so each row is a feature and columns are time frame
    # and gets mean across time frames
    mfccsmean = np.mean(mfccs.T, axis=0)

    return mfccsmean


# Iterate through each sound file and extract the features, uses metadata csv file

# array to hold MFCC data and class labels
features = []
for index, row in metadata.iterrows():
    # get audio file name from csv file column
    file_name = os.path.join(os.path.abspath(
        audio_files), str(row["slice_file_name"]))

    # get class, if the classID is 3 then it is a dark bark
    # set the class label to 1 else not a bark set to 0
    if row["classID"] == 3:
        class_label = 1
    else:
        class_label = 0
    # get mfccs of the audio file
    data = extract_features(file_name)
    # append mfcc data and label to the features array
    features.append([data, class_label])

# Convert features array into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

print("Our data is now in this format:")
print(featuresdf.head())

# Convert features and corresponding classification labels into numpy arrays
features = np.array(featuresdf.feature.tolist())
print(features.size)
labels = np.array(featuresdf.class_label.tolist())
print(labels.size)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build a deep learning model

# convert labels into hot encoded
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)

model = models.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape=(40,)))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

best_model_weights = './model/base.model'
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor='accuracy',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=False,
    save_freq=80
)

callbacks = [checkpoint]
model.compile(optimizer='adam',
              loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(
    X_train,
    train_labels,
    validation_data=(X_test, test_labels),  # use validation set for validation data
    epochs=200,
    verbose=1,
    callbacks=callbacks,
)

print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label="training accuracy")
plt.plot(epochs, val_acc, 'r', label="validation accuracy")
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

model.save_weights('./model/base.model2/dog_model_weights.h5')
model.save('./model/base.model2/dog_model.h5')

