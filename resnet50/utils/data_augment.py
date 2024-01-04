import tensorflow as tf
import tensorflow_addons as tfa
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D, Input
from tensorflow.keras.models import Model

#becuase other method for gaussian blur failed to work, we adopt the following functions

"""
credit for gauss2D() and gaussFilter() to
https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
"""
def gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussFilter(kernel_size=3, sigma=1.0, in_channels=3):
    kernel_weights = gauss2D(shape=(kernel_size, kernel_size), sigma=sigma)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)

    inp = Input(shape=(None, None, in_channels))
    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')(inp)
    model_network = Model(inputs=inp, outputs=g_layer)
    model_network.layers[1].set_weights([kernel_weights])
    model_network.trainable = False
        
    return model_network

#here we generate augmented data as required in paper
class SimCLRDataGenerator:
    def __init__(self, data_gen, batch_size=512, blur_kernel_size=3, blur_sigma=1.0):
            self.data_gen = data_gen
            self.batch_size = batch_size
            self.gauss_blur_model = gaussFilter(kernel_size=blur_kernel_size, sigma=blur_sigma)

    # here as some augmented method comes with a possibility, we create the following function to inplement that
    @staticmethod
    def random_apply(func, x, p):
        """Apply function func to x with probability p."""
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x
        )
    #the following functions are based on the pseudo code in the appendix of the article
    @staticmethod
    def color_distortion(image, s=1.0):
        """Apply color distortion to image."""
        def color_jitter(x):
            x = tf.image.random_brightness(x, max_delta=0.3*s)
            x = tf.image.random_contrast(x, lower=1-0.3*s, upper=1+0.7*s)
            x = tf.image.random_saturation(x, lower=1-0.3*s, upper=1+0.7*s)
            x = tf.image.random_hue(x, max_delta=0.2*s)
            x = tf.clip_by_value(x, 0, 1)
            return x

        def color_drop(x):
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
            return x

        image = SimCLRDataGenerator.random_apply(color_jitter, image, p=0.8)
        image = SimCLRDataGenerator.random_apply(color_drop, image, p=0.2)
        return image

    def custom_augment(self, image):
        """Apply custom augmentations to the image."""
        image = tf.image.random_crop(image, size=[20, 20, 3])
        image = tf.image.resize(image, [32, 32])
        image = SimCLRDataGenerator.random_apply(tf.image.random_flip_left_right, image, p=0.5)
        image = SimCLRDataGenerator.random_apply(tf.image.random_flip_up_down, image, p=0.5)
        image = SimCLRDataGenerator.color_distortion(image, s=1.0)
        image = self.gauss_blur_model(tf.expand_dims(image, 0))[0]
        return image

    def generate(self):
        """Yield a batch of data for SimCLR with two augmented versions of each image."""
        for images in self.data_gen:
            augmented_images_1 = []
            augmented_images_2 = []
            for j in range(images.shape[0]):
                augmented_image_1 = self.custom_augment(images[j])
                augmented_image_2 = self.custom_augment(images[j])
                augmented_images_1.append(augmented_image_1)
                augmented_images_2.append(augmented_image_2)
            yield (tf.stack(augmented_images_1, axis=0), tf.stack(augmented_images_2, axis=0))

    def show_augmented_images(self, num_pairs=5):
        """Displays pairs of augmented images from the generator."""
        augmented_images_1, augmented_images_2 = next(self.generate())

        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 2 * num_pairs))
        for i in range(num_pairs):
            image_1 = tf.cast(augmented_images_1[i], tf.float32)
            image_2 = tf.cast(augmented_images_2[i], tf.float32)
            image_1 = (image_1 - tf.reduce_min(image_1)) / (tf.reduce_max(image_1) - tf.reduce_min(image_1))
            image_2 = (image_2 - tf.reduce_min(image_2)) / (tf.reduce_max(image_2) - tf.reduce_min(image_2))
            image_1 = image_1.numpy()
            image_2 = image_2.numpy()
            axes[i, 0].imshow(image_1)
            axes[i, 0].axis('off')
            axes[i, 0].set_title('Augmented Image 1')
            axes[i, 1].imshow(image_2)
            axes[i, 1].axis('off')
            axes[i, 1].set_title('Augmented Image 2')
        plt.tight_layout()
        plt.show()