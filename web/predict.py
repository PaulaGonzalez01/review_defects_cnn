import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16
from keras.applications import densenet
from keras.applications import inception_v3
from keras.applications import mobilenet_v2

apple_acc = {'vgg16': 82, 'inceptionv3': 93,
             'densenet121': 93, 'mobilenetv2': 98}
mango_acc = {'vgg16': 75, 'inceptionv3': 94,
             'densenet121': 93, 'mobilenetv2': 95}


def predict_one(fruit, model, img_data):
    acc = 0
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

    if fruit == 'apple':
        acc = apple_acc.get(model)
    else:
        acc = mango_acc.get(model)

    if classes[0][0] > 0.5:
        return ("HAS DEFECT", acc)
    elif classes[0][1] > 0.5:
        return ("FRESH", acc)
    return ("Error", 0.0)
