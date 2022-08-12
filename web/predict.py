import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input


def predict_one(fruit, model, img_data):

    model_h5 = keras.models.load_model(f'./models/{fruit}_{model}.h5')
    img = image.image_utils.load_img(img_data, target_size=(256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    classes = model_h5.predict(preprocess_input(img_array))
    if classes[0][0] > 0.5:
        return ("FRESH", classes[0][0])
    elif classes[0][1] > 0.5:
        return ("HAS DEFECT", classes[0][1])
    return ("Error", 0.0)
