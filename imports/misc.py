import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import *


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


def sob_weight(shape, srange=[-360, 360], beta=0.5):
    ret = np.zeros(shape)
    for i in range(shape[0]):
        ret[i, ...] = (1 + np.linspace(srange[0], srange[1], shape[1])**2)**beta
    return ret


def sob_loss(y_true, y_pred, srange=[-360, 360], beta=0.5):
    y = tf.cast(y_true - y_pred, tf.complex64)
    y = tf.squeeze(y)
    y = tf.fft(y)
    # FFT shift for correct multiplication
    y = tf.roll(y, shift=int(y_true.shape[-2]//2), axis=-1)
    y = tf.square(tf.abs(y))
    s = sob_weight(shape=tf.keras.backend.int_shape(y_true)[1:3], srange=srange, beta=beta)
    s = tf.constant(s, tf.float32)
    y = y*s
    return tf.reduce_mean(y)


def _mark_inset(parent_axes, inset_axes, **kwargs):
    # This code is copied from the matplotlib source code and slightly modified.
    # This is done to avoid the 'connection lines'.
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    if 'fill' in kwargs:
        pp = BboxPatch(rect, **kwargs)
    else:
        fill = bool({'fc', 'facecolor', 'color'}.intersection(kwargs))
        pp = BboxPatch(rect, fill=fill, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=1, **kwargs)
    p1.set_clip_on(False)

    p2 = BboxConnector(inset_axes.bbox, rect, loc1=1, **kwargs)
    p2.set_clip_on(False)

    return pp, p1, p2


def zoomed_plot(x, xlim, ylim, zoom=2, text=None, textloc=[], fsize=18, cmap='bone'):

    # This function allows one to create plots with "zoomed in" windows.
    # The rectangle where one desires to zoom in is given using the xlim and ylim arguments.
    # xlim and ylim should contain pixel values, e.g. if we haven an image of size 512 x 512 then
    # xlim = [100, 150] and ylim = [100, 150] shows a zoomed in version of the pixels at locations in xlim and ylim.

    color = 'orange'
    fig, ax = plt.subplots()
    ax.imshow(np.flipud(x), cmap=cmap, origin="lower")
    ax.axis('off')

    axins = zoomed_inset_axes(ax, zoom, loc=4)

    axins.set_xlim(xlim[0], xlim[1])
    axins.set_ylim(ylim[0], ylim[1])

    _mark_inset(ax, axins, fc='none', ec=color)

    axins.imshow(np.flipud(x), cmap=cmap, origin="lower")
    axins.patch.set_edgecolor(color)
    axins.patch.set_linewidth('3')
    axins.set_xticks([], [])
    axins.set_yticks([], [])
    # axins.axis('off')

    if not (text is None):
        ax.text(textloc[0], textloc[1], text, color=color, fontdict={'size': fsize}, transform=ax.transAxes)
    pass


class DataGenerator:
    def __init__(self, operator, operator_up=None, path=data_path, sigma=0.02):
        self._path = path
        self._operator = operator
        self._operator_up = operator_up
        self._sigma = sigma

    def get_batch(self, batch_size=32, mode='train', rescale=1000.):
        p = data_path + mode
        if batch_size is None:
            files = os.listdir(p)
        else:
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
        if batch_size is None:
            files = os.listdir(p)
        else:
            files = np.random.choice(os.listdir(p), size=batch_size, replace=False)
        X = [np.load(p + '/' + file) for file in files]
        y_true = [self._operator_up(x / rescale) for x in X]
        y_noisy = [self._operator(x / rescale) for x in X]
        y_noisy = [y + np.random.normal(0, 1, y.shape) * self._sigma for y in y_noisy]

        y_noisy = np.stack(y_noisy)[..., None]
        y_true = np.stack(y_true)[..., None]
        x_true = np.stack(X)[..., None] / rescale
        return y_noisy, y_true, x_true

