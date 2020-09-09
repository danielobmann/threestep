import os
import tensorflow as tf
import numpy as np


data_path = "../../data/mayoclinic/data/full3mm/"


def cosine_decay(epoch, total, initial=1e-3):
    return initial/2.*(1 + np.cos(np.pi*epoch/total))


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(x_result, x_true, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)

        return 20 * log10(maxval) - 10 * log10(mse)


def PSRN_numpy(x_result, x_true):
    maxval = np.amax(x_true) - np.amin(x_true)
    mse = np.mean((x_true - x_result)**2)
    return 20*np.log10(maxval) - 10*np.log10(mse)


def NMSE(x_result, x_true, name='nmse'):
    with tf.name_scope(name):
        error = tf.reduce_sum((x_result - x_true) ** 2, axis=[1, 2, 3])
        normalizer = tf.reduce_sum(x_true**2, axis=[1, 2, 3])
        return tf.reduce_mean(error/normalizer)


def NMSE_numpy(x_result, x_true):
    error = np.mean((x_true - x_result)**2)
    normalizer = np.mean(x_true**2)
    return error/normalizer


class DataGenerator:
    def __init__(self, operator, operator_up=None, path=data_path, sigma=0.02):
        self._path = path
        self._operator = operator
        self._operator_up = operator_up
        self._sigma = sigma

    def get_batch(self, batch_size=32, mode='train', rescale=1000.):
        p = data_path + mode
        files = np.random.choice(os.listdir(p), size=batch_size, replace=False)
        X = [np.load(p + '/' + file) for file in files]
        y_true = [self._operator(x / rescale) for x in X]
        y_noisy = [y + np.random.normal(0, 1, y.shape) * self._sigma for y in y_true]

        # Bring to correct format for tensorflow
        y_noisy = np.stack(y_noisy)[..., None]
        y_true = np.stack(y_true)[..., None]
        x_true = np.stack(X)[..., None] / rescale

        return y_noisy, y_true, x_true

    def get_batch_upsampling(self, batch_size=32, mode='train', rescale=1000.):
        p = data_path + mode
        files = np.random.choice(os.listdir(p), size=batch_size, replace=False)
        X = [np.load(p + '/' + file) for file in files]
        y_true = [self._operator_up(x / rescale) for x in X]
        y_noisy = [self._operator(x / rescale) for x in X]
        y_noisy = [y + np.random.normal(0, 1, y.shape) * self._sigma for y in y_noisy]

        y_noisy = np.stack(y_noisy)[..., None]
        y_true = np.stack(y_true)[..., None]
        x_true = np.stack(X)[..., None] / rescale
        return y_noisy, y_true, x_true

