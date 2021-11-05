
import h5py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import normalize
from sklearn import svm

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

# Using grid search to train Svm model for classification
param_grid = {'C':[1,10,100],'gamma':[0.01,0.1,0.5,1,1.5,2,5,10],'kernel':['rbf']}
final_score = 0 
final_param = dict()
for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        fold_acc_list = list()
        for fold_index in range(1,6):
            X_train = best_features[meta_data[:,0] != fold_index]
            Y_train = meta_data[meta_data[:,0] != fold_index][:,1]
            r_type = rel_types[meta_data[:,0] == fold_index]
            X_test = best_features[meta_data[:,0] == fold_index]
            # X_test = X_test[r_type == 0]
            Y_test = meta_data[meta_data[:,0] == fold_index][:,1]
            # Y_test = Y_test[r_type == 0]
            clf = svm.SVC(C=C,gamma=gamma)
            clf.fit(X_train,Y_train)
            score = clf.score(X_test,Y_test)
            fold_acc_list.append(score)
            print(f'[CV {fold_index}/5] END ......C={C}, gamma={gamma};, score={score}')
            print(Y_test.shape)

        fold_score = sum(fold_acc_list)/5
        if fold_score > final_score:
            final_score = fold_score
            final_param['C'] = C
            final_param['gamma'] = gamma

# print best train accuracy and Svm parameter
print(final_score)
print(final_param)

