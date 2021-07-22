# TODO: Potentially add license.
# pylint: disable=unexpected-keyword-arg
""" Create tf.data.Dataset from generator """

import tensorflow as tf
import tensorflow_addons as tfa

TF_AUTO = -1  # replace with tf.data.AUTOTUNE for newer versions of tensorflow
AUG_PROBABILITY = 0.3


@tf.function
def _image_scale(im):
    """Scales image from [0,255] to [0,1] and changes dtype to tf.float32.
    
    Args:
        batch: tf.Tensor.
            3D or 4D tensor. 3D for single image, 4D for batch of images.

    Returns: Rescaled float32 tf.Tensor.      
    """
    im = tf.cast(im, tf.float32)
    im = im / 255.0

    return im


@tf.function
def _random_apply(func, im, p, func_kwargs=None):
    """Helper function to randomly apply an augmentation function to data.
    
    Args:
        func: Function.
            Ideally tf.function compatible, otherwise use tf.py_function. 
        im: tf.Tensor.
            3D tensor of single example.
        p: Float.
            Probability threshold for whether function will be applied or not. 
        func_kwargs: kwargs.
            Keyword arguments to be passed to 'func'.

    Returns: Augmented or un-altered version of 'im'.      
    """
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) < p:
        if func_kwargs:
            return func(im, **func_kwargs)
        return func(im)
    return im


@tf.function
def _center_crop(im):
    """ Takes center crop of image to fit 224x224 imagenet dimensions.
    
    Args:
        im: tf.Tensor.
            3D or 4D tensor. 3D for single image, 4D for batch of images.

    Returns: 3D or 4D tensor with height and width dimension being 224x224.    
    """
    OFFSET = 112  # for 224x224 input (112 = 224/2)
    HEIGHT = 1914
    WIDTH = 654
    VERT_CENTER = HEIGHT // 2
    HORIZ_CENTER = WIDTH // 2
    im = tf.image.crop_to_bounding_box(im,
                                       offset_height=VERT_CENTER - OFFSET,
                                       offset_width=HORIZ_CENTER - OFFSET,
                                       target_height=2 * OFFSET,
                                       target_width=2 * OFFSET)

    return im


@tf.function
def _gaussian_noise(im):
    """ Applies gaussian noise to input. 
    
    Args:
        im: tf.Tensor.
            3D tensor of single example.

    Returns: 3D image tensor with added noise. Values are clipped to range [0,1].  
    """
    noise = tf.random.normal(shape=tf.shape(im),
                             mean=(10) / (255),
                             stddev=(20) / (255),
                             dtype=tf.float32)
    noise_im = im + noise
    noise_im = tf.clip_by_value(noise_im, 0.0, 1.0)
    return noise_im


@tf.function
def _cutout_helper(im):
    """ Helper for _cutout function. Applies cutout a single time.
    
    Args:
        im: tf.Tensor.
            3D tensor of single example.

    Returns: 3D image tensor with patch of image zeroed out.  
    """
    # cutout patch can cover max 1/3 of original image
    MX_HEIGHT = 319  # 638/2
    MX_WIDTH = 109  # 218

    mask_height = tf.random.uniform([],
                                    minval=6,
                                    maxval=MX_HEIGHT,
                                    dtype=tf.dtypes.int32)

    mask_width = tf.random.uniform([],
                                   minval=2,
                                   maxval=MX_WIDTH,
                                   dtype=tf.dtypes.int32)

    # must be even numbers
    mask_height = mask_height * 2
    mask_width = mask_width * 2

    return tfa.image.cutout(im, mask_size=(mask_height, mask_width))


@tf.function
def _cutout(im):
    """ Applies cutout (arxiv: 1708.04552) to images. Randomly creates up to 3 patches.
    
    Args:
        im: tf.Tensor.
            3D tensor of single example.

    Returns: 3D image tensor with patch(es) of image zeroed out.  
    """
    # add pseudo batch axis for tfa.cutout
    orig_shape = tf.shape(im)
    im = tf.expand_dims(im, axis=0)

    # Tensorflow has no native for-loop, this can probably be re-written more elegantly though
    tmp = tf.random.uniform([], minval=0, maxval=2, dtype=tf.dtypes.int32)

    im = _cutout_helper(im)
    if tmp > 0:
        im = _cutout_helper(im)
    if tmp > 1:
        im = _cutout_helper(im)

    # remove pseudo batch axis
    im = tf.reshape(im, shape=orig_shape)

    return im


@tf.function
def _image_augment(im):
    """ Passes input through augmentation pipeline for model training.
    
    Args:
        im: tf.Tensor.
            3D tensor of single example.

    Returns: Scaled and augmented 3D image tensor.  
    """
    im = _image_scale(im)

    # gaussian blur
    kernel_size = tf.random.uniform([],
                                    minval=3,
                                    maxval=5,
                                    dtype=tf.dtypes.int32)
    gb_kwargs = {
        'filter_shape': (kernel_size, kernel_size),
    }
    im = _random_apply(tfa.image.gaussian_filter2d,
                       im,
                       p=AUG_PROBABILITY,
                       func_kwargs=gb_kwargs)

    # random rotate
    # angle is in radians
    rot_angle = tf.random.uniform([],
                                  minval=-0.17,
                                  maxval=0.17,
                                  dtype=tf.dtypes.float32)
    rotate_kwargs = {'angles': rot_angle}
    im = _random_apply(tfa.image.rotate,
                       im,
                       p=AUG_PROBABILITY,
                       func_kwargs=rotate_kwargs)

    # gaussian noise
    im = _random_apply(_gaussian_noise, im, p=AUG_PROBABILITY)

    # cutout
    im = _random_apply(_cutout, im, p=AUG_PROBABILITY)

    # random crop
    im = tf.image.random_crop(im, size=(224, 224, 2))

    return im


def _process_aug(inp, lbl):
    im = inp['img_0']
    im = _image_augment(im)

    inp['img_0'] = im

    return inp, lbl


def _process(inp, lbl):
    im = inp['img_0']
    im = _image_scale(im)
    im = _center_crop(im)

    inp['img_0'] = im

    return inp, lbl


def get_ds_from_gen(generator,
                    out_mode='combined',
                    mode='test',
                    batch_size=64,
                    cache_dir=''):
    """ Creates tf.data.Dataset from generator.
    
    Args:
        generator: python generator.
            tf.keras.utils.Sequence or generic python generator are acceptable output batch size should be 1. 
            Tested only with utils.csv_generator.
        out_mode: One of {'combined', 'meta', 'image'}.
            Default is 'combined'.
            What part of the dataset to return.
        mode: One of {'train', 'valid', 'test'}.
            Default is 'test'.
            Determines whether augmentations are applied and what kind of cropping is used to fit images to 224x224 shape. 
        batch_size: Int.
            Batches returned by dataset iterator. 
        cache_dir: Filename.
            Where to cache dataset. It is recommended to provide a filepath here for larger datasets or remove caching.
            For no filename, dataset will be cached in memory. For only a single pass over data this provides no speedup.

    Returns: tf.data.Dataset instance.  
    """
    if out_mode == 'combined':
        ds = tf.data.Dataset.from_generator(lambda: generator,
                                            output_types=({
                                                'img_0': tf.float32,
                                                'meta_0': tf.float32
                                            }, tf.float32),
                                            output_shapes=({
                                                'img_0': (1914, 654, 2),
                                                'meta_0': (22)
                                            }, (1)))

    if out_mode == 'meta':
        ds = tf.data.Dataset.from_generator(lambda: generator,
                                            output_types=({
                                                'meta_0': tf.float32
                                            }, tf.float32),
                                            output_shapes=({
                                                'meta_0': (22)
                                            }, (1)))

    if out_mode == 'image':
        ds = tf.data.Dataset.from_generator(lambda: generator,
                                            output_types=({
                                                'img_0': tf.float32
                                            }, tf.float32),
                                            output_shapes=({
                                                'img_0': (1914, 654, 2)
                                            }, (1)))

    # can customize to optimize performance for your system (see: https://www.tensorflow.org/guide/data_performance)
    prefetch_sample_num = 10 * batch_size

    if mode == 'train':
        if out_mode == 'meta':
            ds = ds.cache(filename=cache_dir).prefetch(
                prefetch_sample_num).batch(batch_size)
        else:
            ds = ds.cache(
                filename=cache_dir).prefetch(prefetch_sample_num).map(
                    _process_aug,
                    num_parallel_calls=TF_AUTO).batch(batch_size).prefetch(5)
    else:
        if out_mode == 'meta':
            ds = ds.prefetch(prefetch_sample_num).batch(batch_size)
        else:
            ds = ds.prefetch(prefetch_sample_num).batch(batch_size).map(
                _process).prefetch(5)

    return ds