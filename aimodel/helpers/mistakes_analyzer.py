import pathlib

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from helpers.custom_functions import weighted_cross_entropy, soft_f1
from helpers.preprocessing import *

model = tf.keras.models.load_model('../piano_model', custom_objects={
        'f1_score': tfa.metrics.F1Score(51, average='micro', threshold=0.00000000000000000000000000000000000000001),
        'weighted_cross_entropy_with_logits': weighted_cross_entropy(10)})

data_dir = pathlib.Path('D:\\VolodymyrRushchak\\projects\\pianocamera\\dataset').with_suffix('')
image_count = len(list(data_dir.glob('*.png')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, seed=1234, reshuffle_each_iteration=False)
val_size = int(image_count * 0.2)
val_ds = list_ds.take(val_size)
image_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
image_ds = configure_for_performance(image_ds)

true_positives = np.zeros(51)
false_positives = np.zeros(51)
false_negatives = np.zeros(51)
for img_batch, label_batch in image_ds:
    predictions = tf.sigmoid(model.predict(img_batch))
    y_pred = tf.cast(tf.greater(predictions, 0.5), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * label_batch, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - label_batch), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * label_batch, axis=0), tf.float32)
    true_positives += tp.numpy()
    false_positives += fp.numpy()
    false_negatives += fn.numpy()

f1_scores = 2 * true_positives / (2 * true_positives + false_negatives + false_positives + 1e-16)
print(f1_scores)




