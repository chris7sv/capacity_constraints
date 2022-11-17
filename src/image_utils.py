import os, glob


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage.io import imread
from tensorflow.keras.datasets import cifar10
from skimage.color import rgb2gray, rgb2grey, gray2rgb
from scipy.ndimage.filters import gaussian_filter


import model_utils as mu



def apply_input_noise(img, noise_lvl, stdev=50):
    """
    Adds a desired amount of gaussian noise

    to an image.

    noise_lvl - [0.-1.]specifies what proportion
                 of the image pixels will be modified
    stdev - the standard deviation of the zero-mean gaussian
    """

    if 1. < noise_lvl <= 0.:
        raise Exception('Invalid value for image noise level')

    noisy_pixels = int(img.size*noise_lvl)
    mask = np.zeros(img.size)
    mask[:noisy_pixels] += 1
    np.random.shuffle(mask)
    mask = mask.astype(bool).reshape(img.shape)

    img[mask] += np.random.uniform(-stdev, stdev,noisy_pixels)

    return img

def sketchify_opencv(image, kernel=3):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    image = cv.GaussianBlur(image, (kernel, kernel), 0)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, delta=delta,
                      borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, delta=delta,
                      borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#    grad = 255 - grad
    recast = cv.cvtColor(grad, cv.COLOR_GRAY2RGB)
    return recast

def salt_and_pepper_noise(image, p):

    assert 0 <= p <= 1

    # image = image.astype('float32')/255.

    image = rgb2gray(image)
    assert image.ndim == 2

    u = np.random.uniform(size=image.shape)

    salt = (u >= 1 - p/2).astype(image.dtype)
    pepper = -(u < p/2).astype(image.dtype)


    image = image + salt + pepper
    image = np.clip(image, 0, 1)

    assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image#np.stack((image,)*3,axis=-1)


def apply_uniform_noise(image, low, high, rng=None):
    """Apply uniform noise to an image, clip outside values to 0 and 1.
    parameters:
    - image: a numpy.ndarray 
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    image = rgb2gray(image)

    nrow = image.shape[0]
    ncol = image.shape[1]


    image = image + get_uniform_noise(low, high, nrow, ncol, rng)

    #clip values
    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)

    assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image


def get_uniform_noise(low, high, nrow, ncol, rng=None):
    """Return uniform noise within [low, high) of size (nrow, ncol).
    parameters:
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - nrow: number of rows of desired noise
    - ncol: number of columns of desired noise
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    if rng is None:
        return np.random.uniform(low=low, high=high,
                                 size=(nrow, ncol))
    else:
        return rng.uniform(low=low, high=high,
                           size=(nrow, ncol))

def high_pass_filter(image, std):

    """Copied from Geirhos et al 2018"""

    """
    Apply a Gaussian high pass filter to a greyscale converted image.
    by calculating Highpass(image) = image - Lowpass(image).

    parameters:
    - image: a numpy.ndarray
    - std: a scalar providing the Gaussian low-pass filter's standard deviation
    """

    # set this to mean pixel value over all images
    bg_grey = 0.4423

    # convert image to greyscale and define variable prepare new image
    image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly the gaussian filter and subtract from the original image
    gauss_filter = gaussian_filter(image, std, mode ='constant', cval=bg_grey)
    new_image = image - gauss_filter

    # add mean of old image to retain image statistics
    mean_diff = bg_grey - np.mean(new_image, axis=(0, 1))
    new_image = new_image + mean_diff

    # crop too small and too large values
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    return np.dstack((new_image, new_image, new_image))

def low_pass_filter(image, std):

    """ Copied from Geirhos et al 2018"""
    """Aplly a Gaussian low-pass filter to an image.

    parameters:
    - image: a numpy.ndarray
    - std: a scalar providing the Gaussian low-pass filter's standard deviation
    """
    # set this to mean pixel value over all images
    bg_grey = 0.4423

    # covert image to greyscale and define variable prepare new image
    image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly Gaussian low-pass filter
    new_image = gaussian_filter(image, std, mode='constant', cval=bg_grey)

    # crop too small and too large values
    new_image[new_image < 0] = 0
    new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    return np.dstack((new_image, new_image, new_image))


def standardize_img(img):
    return (img - np.mean(img))/np.std(img)

def standardize_data(data):
    for i in range(len(data)):
        data[i] = standardize_img(data[i])
    return data

def standardize_data_v2(data):
    data = data/127.5
    x -= 1.
def clip_extremes(data, least=0, most=255):
    data[data < least] = least
    data[data > most] = most
    return data


def grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def dodge(front, back):
    result = front*255/(255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')

def sketchify_image(img):
    gray_img = grayscale(img)
    inverted_img = 255 - gray_img
    blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=5)
    final_img = dodge(blur_img, gray_img)
    channeled = np.stack((final_img,)*3, axis=-1)
    return channeled

def is_in_bounds(mat, low=0, high=1):
    """Return wether all values in 'mat' fall between low and high.
    parameters:
    - mat: a numpy.ndarray
    - low: lower bound (inclusive)
    - high: upper bound (inclusive)
    """

    return np.all(np.logical_and(mat >= low, mat <= high))

def load_and_prep_data(gen_type, noise_type, n_val, standardize=True,
        save=True):

    # Load data
    if gen_type != 'sketches':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Scale responses to 0-1


    if gen_type == 'noisy':
        x_train = x_train/255.
        x_test = x_test/255.
        # Apply image noise
        print(f'Using {noise_type} input noise.')
        for idx, img in enumerate(x_train):
            if noise_type == 'uniform':
                x_train[idx] = gray2rgb(apply_uniform_noise(img, -n_val, n_val))
            else:
                x_train[idx] = gray2rgb(salt_and_pepper_noise(img, n_val))
        for idx, img in enumerate(x_test):
            if noise_type == 'uniform': 
                x_test[idx] = gray2rgb(apply_uniform_noise(img, -n_val, n_val))
            else:
                x_test[idx] = gray2rgb(salt_and_pepper_noise(img, n_val))
    elif gen_type == 'sketch':
        # Use sobel filter to trace edges
        for idx, img in enumerate(x_train):
            x_train[idx] = sketchify_opencv(img)
        for idx, img in enumerate(x_test):
            x_test[idx] = sketchify_opencv(img)
        #Scale responses to 0-1
        x_train = x_train/255.
        x_test = x_test/255.
        #needed since some values would exceed 255 / 1 when noise is added
        x_train = clip_extremes(x_train, most=1)
        x_test = clip_extremes(x_test, most=1)
        # Save some examples for visualization
    elif gen_type == 'none':
        x_train = x_train/255.
        x_test = x_test/255.
    elif gen_type == 'high':
        x_train = x_train/255.
        x_test = x_test/255.
        for idx, img in enumerate(x_train):
            x_train[idx] = high_pass_filter(img, n_val)
        for idx, img in enumerate(x_test):
            x_test[idx] = high_pass_filter(img, n_val)
    elif gen_type == 'low':
        x_train = x_train/255.
        x_test = x_test/255.
        for idx, img in enumerate(x_train):
            x_train[idx] = low_pass_filter(img, n_val)
        for idx, img in enumerate(x_test):
            x_test[idx] = low_pass_filter(img, n_val)
    elif 'ood' in gen_type:
        cond = 'real' if 'real' in gen_type else 'draw'#'drawing'
        files = [pic for pic in \
                 glob.glob('/home/gx19122/Pictures/out_of_domain_test/horses/*') \
                 if not os.path.isdir(pic)]
        files_cond = [pic for pic in files if '{}'.format(cond) in pic]
        x_test = np.zeros(shape=(len(files_cond), 32, 32, 3))
        for idx, f in enumerate(files_cond): 
            img = imread(f)
            if len(img.shape) < 3: 
                img = np.dstack((img, img, img)) 
            x_test[idx] = img
        y_test = np.zeros(len(files_cond), dtype='uint8') + 7 # ??? 

        x_test /= 255.
        x_train = x_test
        y_train = y_test
    if save:
        for i in range(10):
            plt.imsave(os.path.join(os.getcwd(),
                                    f'cifar_{gen_type}_{noise_type}_{n_val}\
                                        _{i}.png'), x_train[i])
        # plt.imsave(os.path.join(os.getcwd(),f'cifar_test_uniform{n_val}_{i}.png'), x_test[i])

    # Convert targets to one-hot
    y_train_copy = np.copy(y_train).reshape(len(y_train),)
    y_test_copy = np.copy(y_test).reshape(len(y_test),)
    y_train = mu.vectorize_labels(y_train, 10)
    y_test = mu.vectorize_labels(y_test, 10)

    #standardize
    if standardize:
        x_train = standardize_data(x_train)
        x_test = standardize_data(x_test)

    return (x_train, y_train, y_train_copy), (x_test, y_test, y_test_copy)

def generate_random_image(num_samples, data_shape,
        rnd, random_type='rnd-gauss', means=None, std=None, data=None):

    if random_type == 'rnd-gauss':

        data = np.ndarray([num_samples,
                           data_shape[0],
                           data_shape[1],
                           data_shape[2]])

        for i in range(num_samples):
            if means is None:
                data[i] = rnd.normal(120.75, 69.93,
                                     (data_shape[0]
                                      *data_shape[1]
                                      *data_shape[2])).reshape(
                                          data_shape[0],
                                          data_shape[1],
                                          data_shape[2])
            elif isinstance(means, (list, tuple)):
                data_point = np.ndarray(shape=data_shape)
                for idx, stat in enumerate(zip(means, std)):
                    data_point[:, :, idx] = rnd.normal(stat[0], stat[1],
                                                       (data_shape[0]
                                                        *data_shape[1])).reshape(
                                                            data_shape[0],
                                                            data_shape[1])
                data[i] = np.copy(data_point)
    elif random_type == 'shuffle-pixels':
        assert type(data) is not None, 'Please specify a data set to shuffle'
        for idx, x in enumerate(data):
            x = x.flatten()
            np.random.shuffle(x)
            x = np.reshape(x, (data_shape))
            data[idx] = x

    else:
        raise NameError('Not implemented yet, please use rnd-gauss noise')
    
#    data = clip_extremes(data)
    return data/255