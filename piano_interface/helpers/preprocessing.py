import tensorflow as tf

from helpers.constants import AIMODEL_HEIGHT, AIMODEL_WIDTH


@tf.function
def preprocess(img: tf.Tensor, piano_boundary: tf.Tensor) -> tf.Tensor:
    xs = piano_boundary[:, 0]
    ys = piano_boundary[:, 1]
    xleft, xright = tf.math.reduce_min(xs), tf.math.reduce_max(xs)
    ytop, ybottom = tf.math.reduce_min(ys), tf.math.reduce_max(ys)
    img = tf.image.crop_to_bounding_box(img, ytop, xleft, ybottom - ytop, xright - xleft)
    return tf.expand_dims(tf.image.resize(img, [AIMODEL_HEIGHT, AIMODEL_WIDTH]), axis=0)
