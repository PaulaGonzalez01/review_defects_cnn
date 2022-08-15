import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16
from keras.applications import densenet
from keras.applications import inception_v3
from keras.applications import mobilenet_v2


def predict_one(fruit, model, img_data):

    model_h5 = keras.models.load_model(f'./models/{fruit}_{model}.h5')
    img = image.image_utils.load_img(img_data, target_size=(256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    classes = []
    if model == 'densenet121':
        classes = model_h5.predict(densenet.preprocess_input(img_array))
    elif model == 'vgg16':
        classes = model_h5.predict(vgg16.preprocess_input(img_array))
    elif model == 'inceptionv3':
        classes = model_h5.predict(inception_v3.preprocess_input(img_array))
    else:
        classes = model_h5.predict(mobilenet_v2.preprocess_input(img_array))

    if classes[0][0] > 0.5:
        return ("FRESH", classes[0][0])
    elif classes[0][1] > 0.5:
        return ("HAS DEFECT", classes[0][1])
    return ("Error", 0.0)
