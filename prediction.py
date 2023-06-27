import os

import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array

model_path = 'WeedClassifier.h5'

model = tf.keras.models.load_model(model_path)

test_image_directory = './dataset/prediction/'

input_shape = (512, 512)

for image_filename in os.listdir(test_image_directory):
    image_path = os.path.join(test_image_directory, image_filename)
    image = load_img(image_path, target_size=input_shape)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    classes = ['weed', 'crop']
    predicted_class = classes[predicted_class_index]

    print(f"Image: {image_filename}, Predicted Class: {predicted_class}")
