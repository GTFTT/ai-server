import tensorflow as tf
import numpy as np

from tensorflow.keras import models
from numpy import asarray


def load_model(modelName = "digits_recognizer_model"):
    # Load model from hard drive
    model = models.load_model(modelName)
    return model

# Takes 28x28 images of pixel values from 0 to 255,
# converts into array of bytes and predicts
# returns predicted digit
def predict_image(image):

    model = load_model()

    image_array = asarray(image, dtype=int)
    # print("Array image: ", type(image_array))
    # print("Shape: ", image_array.shape)# summarize shape

    my_image = image_array
    my_list = my_image.tolist() # Image converted to array(list)

    # Invert color(array values)
    for c in range(len(my_list)):
        for x in range(len(my_list[c])):
            my_list[c][x] = 255-my_list[c][x]
    # print("My list: ", my_list)

    # print()
    # print("My image: ", type(my_image))
    # print("My image: ", type(my_image.tolist()))
    # print("My image: ", my_image.tolist())
    # print('\nShape: ', np.shape(my_image))
    # print('DType: ', my_image[0][0], type(my_image[0][0]))
    # plt.imshow(x_train[0], cmap=plt.cm.binary)

    npy_images_arr = np.ndarray((1, 28, 28),
                                buffer=np.array(my_list), # Convert numpy array to list and makes it flat for using buffered values
                                # buffer=np.array(my_image.tolist()), # Convert numpy array to list and makes it flat for using buffered values
                                dtype=int)

    #Make predictions
    predictions = model.predict(npy_images_arr)
    # print("Predictions: ", predictions[0])
    print("Norm predictions: ", np.argmax(predictions[0]))
    return np.argmax(predictions[0])