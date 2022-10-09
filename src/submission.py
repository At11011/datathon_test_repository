import numpy as np
from glob import glob
from PIL import Image, ImageOps
from itertools import permutations
from keras.models import load_model
import tensorflow as tf
import os

import utils

class Predictor:
    """
    DO NOT RENAME THIS CLASS
    This class enables automated judging
    This class should stay named as `Predictor`
    """

    def __init__(self):
        """
        Initializes any variables to be used when making predictions
        """
        self.model = load_model('unscramble_model.h5')
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def make_prediction(self, img_path):
        """
        DO NOT RENAME THIS FUNCTION
        This function enables automated judging
        This function should stay named as `make_prediction(self, img_path)`

        INPUT:
            img_path: 
                A string representing the path to an RGB image with dimensions 128x128
                example: `example_images/1.png`
        
        OUTPUT:
            A 4-character string representing how to re-arrange the input image to solve the puzzle
            example: `3120`
        """

        # Load the image
        # Grayscale the image. Our model determines whether a grayscale image is correctly arranged or not
        img = ImageOps.grayscale(Image.open(f'{img_path}'))

        # Converts the image to a 3D numpy array (128x128x3)
        img_array = np.asarray(img, dtype=np.uint8)/255 # Normalize

        predictions = np.zeros(24)
        
        # Generate all permutations
        for i in range(24):
            img = utils.image_perm(img_array, i)
            img = (np.expand_dims(img,0))
            # print(img.shape)
            single_prediction = self.probability_model.predict(img, verbose='false')
            # print(single_prediction)
            predictions[i] = single_prediction[0][1]
        
        idx = np.argmax(predictions)
        return f'{utils.perms[idx][0]:d}' + f'{utils.perms[idx][1]:d}' + f'{utils.perms[idx][2]:d}' + f'{utils.perms[idx][3]:d}' # f{""}

# Run this file using `python3 submission.py`
if __name__ == '__main__':

    direc = "example_images/*"
    direc = "../assets/testing/*"
    # direc = "../assets/train/1032/*"
    count = 0
    for img_name in glob(direc):
        
        # Open an example image using the PIL library
        example_image = Image.open(img_name)

        # Use instance of the Predictor class+--- to predict the correct order of the current example image
        predictor = Predictor()
        prediction = predictor.make_prediction(img_name)

        print(prediction)

        # Visualize the image
        pieces = utils.get_uniform_rectangular_split(np.asarray(example_image), 2, 2)
        
        a = int(prediction[0])
        b = int(prediction[1])
        c = int(prediction[2])
        d = int(prediction[3])
        final_image = Image.fromarray(np.vstack((np.hstack((pieces[a],pieces[b])),np.hstack((pieces[c],pieces[d])))))
        final_image.show()
        
        count += 1

        if count >= 10:
            break