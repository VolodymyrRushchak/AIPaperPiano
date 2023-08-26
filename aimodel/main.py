import pathlib
import pickle

from constants import *
from helpers.preprocessing import process_path, configure_for_performance
from helpers.custom_functions import weighted_cross_entropy
import tensorflow_addons as tfa


feature_extractor = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
)
feature_extractor.trainable = False
# print(feature_extractor.summary())

cropped_feature_extractor = tf.keras.models.Sequential(feature_extractor.layers[:-8])
inpt = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
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
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[1100, 4200], values=[0.0001, 0.00001, 0.000003])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=weighted_cross_entropy(10),
              metrics=[tf.keras.metrics.Recall(0.), tf.keras.metrics.Precision(0.),
                       tfa.metrics.F1Score(51, average='micro', threshold=0.000000000000000000000000000000000000001)])

print(cropped_feature_extractor.summary())
print(model.summary())
print(tf.config.experimental.get_memory_info('GPU:0')['current'])

data_dir = pathlib.Path('D:\\VolodymyrRushchak\\projects\\pianocamera\\dataset').with_suffix('')
image_count = len(list(data_dir.glob('*.png')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, seed=1234, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

save_callback = tf.keras.callbacks.ModelCheckpoint(
    'ai_model', monitor='val_f1_score', save_best_only=True, mode='max', verbose=1, save_weights_only=True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=150,
    callbacks=[save_callback]
)

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
