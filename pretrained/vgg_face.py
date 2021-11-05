import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation
import tensorflow.keras.backend as K



# Build VGG_FACE_MODEL
def vgg_face():
    # Define VGG_FACE_MODEL architecture
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    # Load VGG Face model weights
    model.load_weights('vgg_face_weights.h5')

    # For extracting FC7 layer, uncomment next five line and
    # comment final line of method

    # model = Model(inputs= model.layers[0].input,outputs= model.layers[-6].output)
    # seq_model = Sequential()
    # seq_model.add(model)
    # seq_model.add(Flatten())
    # return seq_model

    return Model(inputs= model.layers[0].input,outputs= model.layers[-2].output)


if __name__ == '__main__':
    vgg_face().summary()