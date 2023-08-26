import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function
def macro_soft_f1(y, y_hat):
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


def soft_f1(y, y_hat) -> tf.Tensor:
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    return soft_f1


def weighted_cross_entropy(weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, weight
        )
        return loss
    return weighted_cross_entropy_with_logits


def f1_function(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    return tp, fp, fn


class F1Metric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f1_fn = f1_function
        self.tp_count = self.add_weight("tp_count", initializer="zeros", dtype=tf.float32)
        self.fp_count = self.add_weight("fp_count", initializer="zeros", dtype=tf.float32)
        self.fn_count = self.add_weight("fn_count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fp, fn = self.f1_fn(y_true, y_pred)
        self.tp_count.assign_add(tf.cast(tp, dtype=tf.float32))
        self.fp_count.assign_add(tf.cast(fp, dtype=tf.float32))
        self.fn_count.assign_add(tf.cast(fn, dtype=tf.float32))

    def result(self):
        precision = self.tp_count / (self.tp_count + self.fp_count)
        recall = self.tp_count / (self.tp_count + self.fn_count)
        f1 = tf.constant(2, dtype=tf.float32) * (precision * recall) / (precision + recall)
        return f1
