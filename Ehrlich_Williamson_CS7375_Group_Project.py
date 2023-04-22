import pandas as pd
import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa.display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# variable to hold number of mfcc samples to take
n_mfcc = 40

# Set the path to the full UrbanSound dataset
audio_files = '/Users/matt/UrbanSound8K/audio/'
# Set path to metadata csv file included with UrbanSound dataset
metadata = pd.read_csv('/Users/matt/UrbanSound8k/UrbanSound8K.csv')

# function to generate mfcc values from audio files


def extract_features(file_name):
    # librosa.load() automatically normalizes audio, converts to mono, and sets sample rate to 22050
    audio, sample_rate = librosa.load(file_name)
    # compute mfccs from audio files. Returns a 2D array where each column corresponds to a time frame and each row is a mfcc
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    # transpose mfccs array so that rows and columns are swapped so each row is a feature and columns are time frame and gets mean across time frames
    mfccsmean = np.mean(mfccs.T, axis=0)

    return mfccsmean


# array to hold MFCC data and class labels
features = []

# Iterate through each sound file and extract the features, uses metadata csv file
for index, row in metadata.iterrows():

    # get audio file name from csv file column
    file_name = os.path.join(os.path.abspath(
        audio_files), str(row["slice_file_name"]))

    # get class of audio file (label)
    class_label = row["class"]
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
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix for all class labels:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report for all class labels:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# ----this section converts dog_bark to 1 and all other labels to 0
# Define the target class label
target_class = 'dog_bark'

# Create new DataFrame that will convert class labels to either dog bark or not dog bark
binaryfeaturesdf = pd.DataFrame(featuresdf['feature'].tolist(), columns=[
    f'MFCC_{i}' for i in range(1, n_mfcc + 1)])
binaryfeaturesdf['class_label'] = featuresdf['class_label']

# Adds column 'binary_label' based on class_label being dog bark or not dog bark
binaryfeaturesdf['binary_label'] = binaryfeaturesdf['class_label'].apply(
    lambda x: "dog_bark" if x == target_class else "not_dog_bark")

# Separate the features (X) and the binary labels (y)
X = binaryfeaturesdf.drop(
    ['class_label', 'binary_label'], axis=1)  # puts MFCCs into X
y = binaryfeaturesdf['binary_label']  # puts binary labels into y

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix for dog barks versus not dog barks:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report for dog barks versus not dog barks:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))


# ---balance the RandomForestClassifier to account for low number of dog barks vs not dog barks
# Create and train the Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix when using balanced class weight for RandomForestClassifier:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report when using balanced class weight for RandomForestClassifier:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Using SMOTE resampling to try to improve accuracy by balancing dog barks vs not dog barks
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train the classifier on the resampled training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_resampled, y_resampled)

# Make predictions on the original test data using the resampled classifier
y_pred = clf.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix when using SMOTE to balance dog barks versus not dog barks:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report when using SMOTE to balance dog barks versus not dog barks:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# The SVC classifier uses a linear kernel and C=1
svc_clf = Pipeline([("scaler", StandardScaler()),
                   ("svc", SVC(kernel="linear", C=1))])
svc_clf.fit(X_train, y_train)
y_pred = svc_clf.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix when using SVC:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report when using SVC:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))


# This section generates a figure showing the confusion matrix after SVC with labels-just a clearer way to view data
class_labels = ['Dog Bark', 'Not Dog Bark']
# Create a heatmap of the confusion matrix with row labels
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels,
            yticklabels=class_labels, cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
