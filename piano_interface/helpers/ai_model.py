import tensorflow as tf
import tensorflow_addons as tfa
from helpers.constants import AIMODEL_HEIGHT, AIMODEL_WIDTH


def weighted_cross_entropy(weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, weight
        )
        return loss
    return weighted_cross_entropy_with_logits


def get_ai_model() -> tf.keras.Model:
    feature_extractor = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(AIMODEL_HEIGHT, AIMODEL_WIDTH, 3)
    )
    feature_extractor.trainable = False

    cropped_feature_extractor = tf.keras.models.Sequential(feature_extractor.layers[:-8])
    inpt = tf.keras.layers.Input(shape=(AIMODEL_HEIGHT, AIMODEL_WIDTH, 3))
    x = tf.keras.layers.RandomBrightness(0.2)(inpt)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = cropped_feature_extractor(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(51, name='predictions', kernel_regularizer=tf.keras.regularizers.L2(l2=0.1))(x)

    model = tf.keras.Model(inpt, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000005), loss=weighted_cross_entropy(10),
                  metrics=[tf.keras.metrics.Recall(0.), tf.keras.metrics.Precision(0.),
                           tfa.metrics.F1Score(51, average='micro',
                                               threshold=0.000000000000000000000000000000000000001)])
    model.load_weights('assets/ai_model/fine_tuned_model')
    return model
