import h5py
import pickle
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Convolution2D, MaxPooling2D, Dense,
                                     Dropout, Softmax, Flatten, Activation,
                                     Subtract,Multiply,Add,Concatenate,Input)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from pretrained.vgg_face import vgg_face
from utils.preprocessor import fisher_score


# Read data from saved h5 file
# parents contains parents images
# child contains children images
# rel_types defines type of relationship between parent and chlid
# meta_data contains pairs labels and fold number
hf = h5py.File('data.h5','r')
p_images = np.array(hf['parents'])
c_images = np.array(hf['child'])
rel_types = np.array(hf['rel_types'])
meta_data = np.array(hf['meta_data'])
hf.close()


# Use vgg_face model as feature extractor
feature_extractor = vgg_face()

# Extract features using vgg-face model
images = np.vstack((p_images,c_images))
tmp = feature_extractor.predict(images)
p_features = tmp[:tmp.shape[0]//2]
c_features = tmp[tmp.shape[0]//2:]

# Merge feature vectors using various functions
# You can uncomment each line to use that function for merging

# features = p_features + c_features
# features = np.abs(p_features - c_features)
# features = (p_features - c_features)**2
# features = np.hstack((p_features,c_features))
# features = np.hstack((p_features + c_features,(p_features - c_features)**2))
features = np.hstack((p_features + c_features,np.abs(p_features - c_features)))


# Apply feature normalization
normalized_features = normalize(features)


# Apply fisher score algorithm and use first 5244 features
# for classification
selected_feature_count = 2622*2 
ranks = fisher_score(np.hstack((meta_data[:,1].reshape((-1,1)),normalized_features)))
tmp_array = np.vstack((ranks,normalized_features))
sorted_features = tmp_array[:,np.argsort(tmp_array[1,:])][1:]
best_features = sorted_features[:,:selected_feature_count] 


# Selecting fold number for data spliting
fold_index=3
X_train = best_features[meta_data[:,0] != fold_index]
Y_train = meta_data[meta_data[:,0] != fold_index][:,1]
X_test = best_features[meta_data[:,0] == fold_index]
Y_test = meta_data[meta_data[:,0] == fold_index][:,1]

# Define classifire architecture
classifire = Sequential()
classifire.add(Dense(1024,activation='relu'))
classifire.add(Dropout(0.5))
classifire.add(Dense(256,activation='relu'))
classifire.add(Dropout(0.5))
classifire.add(Dense(64,activation='relu'))
classifire.add(Dropout(0.5))
classifire.add(Dense(1,activation='sigmoid'))

# Compile and train classifire
classifire.compile(loss='binary_crossentropy', metrics=['acc'], optimizer= Adam())
history = classifire.fit(x=X_train, y=Y_train, validation_data=(X_test,Y_test), batch_size=64, epochs=50)

# Plot loss function for train and test set
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show() 

# Plot accuracy for train and test set
plt.title("Fold 1")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.plot(history.history['acc'],label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.legend()
plt.show()