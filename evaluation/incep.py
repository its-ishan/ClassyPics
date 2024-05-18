import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import cv2
import os


def calculate_inception_score(images, num_batches=10):
    inception_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
                                                        input_shape=(299, 299, 3))
    def preprocess_image(image):
        image = tf.image.resize(image, (299, 299))
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image

    preprocessed_images = np.array([preprocess_image(image) for image in images])

    activations = inception_model.predict(preprocessed_images)

    softmax_activations = tf.nn.softmax(activations)

    marginal_distribution = tf.reduce_mean(softmax_activations, axis=0)

    kl_divergences = [tf.keras.losses.KLDivergence()(marginal_distribution, softmax_activations[i])
                      for i in range(len(softmax_activations))]

    inception_score = tf.exp(tf.reduce_mean(kl_divergences))

    return inception_score.numpy()

def read_images_from_folder(folder_path, target_size=(299, 299)):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    images = []
    for file in image_files:
        file_path = os.path.join(folder_path, file)
        # Read image using OpenCV
        image = cv2.imread(file_path)
        # Resize image if necessary
        if target_size:
            image = cv2.resize(image, target_size)
        images.append(image)
    return images

folder_path = '/mnt/nvme0n1p5/projects/hackathon/CP2/tools/mnist/cond_class_samples/1'
images = read_images_from_folder(folder_path)


images_array = np.array(images)
#
# is_score = calculate_inception_score(images_array)
# print("Inception Score:", is_score)
