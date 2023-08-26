import os
from constants import *


def get_label(file_name):
    pressed_keys_str = tf.strings.split(file_name, '#')[0]
    pressed_keys = tf.strings.to_number(tf.strings.split(pressed_keys_str, '_'), out_type=tf.int32)
    return tf.keras.layers.CategoryEncoding(num_tokens=51, output_mode="multi_hot")(pressed_keys)


def decode_img(img, file_name):
    img = tf.io.decode_png(img, channels=3)
    piano_coordinates_str = tf.strings.split(tf.strings.split(file_name, '@')[0], '#')[1]
    piano_coordinates = tf.strings.to_number(tf.strings.split(piano_coordinates_str, '_'), out_type=tf.int32)
    xs = piano_coordinates[::2]
    ys = piano_coordinates[1::2]
    xleft, xright = tf.math.reduce_min(xs), tf.math.reduce_max(xs)
    ytop, ybottom = tf.math.reduce_min(ys), tf.math.reduce_max(ys)
    img = tf.image.crop_to_bounding_box(img, ytop, xleft, ybottom - ytop, xright - xleft)
    return tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])


def process_path(file_path):
    file_name = tf.strings.split(file_path, os.path.sep)[-1]
    label = get_label(file_name)

    img = tf.io.read_file(file_path)
    img = decode_img(img, file_name)
    return img, label


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
