import os
import json

import tensorflow as tf

with open('config.json') as config_file:
    model_type = json.load(config_file)['model_type']

os.mkdir(str(model_type))


def normalize_01(image):
    return tf.cast(image, tf.float32) / 255


def downscale(image):
    return tf.image.resize(image, (270, 270), tf.image.ResizeMethod.AREA)


train = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    labels=None, label_mode=None,
    batch_size=12, image_size=(1080, 1080),
    validation_split=0.2, subset='training',
    seed=1
).map(lambda x: (
    normalize_01(tf.image.rgb_to_grayscale(downscale(x))),
    normalize_01(tf.image.rgb_to_grayscale(x))
)).prefetch(24)

valid = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    labels=None, label_mode=None,
    batch_size=12,
    image_size=(1080, 1080),
    validation_split=0.2, subset='validation',
    seed=1
).map(lambda x: (
    normalize_01(tf.image.rgb_to_grayscale(downscale(x))),
    normalize_01(tf.image.rgb_to_grayscale(x))
)).prefetch(24)

kwargs = {
    'padding': 'same',
    'activation': tf.keras.activations.relu,
    'kernel_initializer': tf.keras.initializers.Orthogonal
}
input_layer = tf.keras.Input(shape=(None, None, 1))
layer = tf.keras.layers.Conv2D(64, 5, **kwargs)(input_layer)
layer = tf.keras.layers.Conv2D(64, 3, **kwargs)(layer)
layer = tf.keras.layers.Conv2D(32, 3, **kwargs)(layer)
layer = tf.keras.layers.Conv2D(16, 3, **kwargs)(layer)
output_layer = tf.nn.depth_to_space(layer, 4)

model = tf.keras.Model(input_layer, output_layer)

if model_type == 'dpsnr':
    def dpsnr(y_true, y_pred):
        return 108.79928 - tf.image.psnr(y_true, y_pred, 1)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=dpsnr)

elif model_type == 'dssim':
    def dssim(y_true, y_pred):
        return 1 - tf.image.ssim(y_true, y_pred, 1)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=dssim)

elif model_type == 'mse':
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.MeanSquaredError())

model.fit(
    train,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.Callback(),
        tf.keras.callbacks.EarlyStopping('loss', patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            f'{model_type}/',
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        )
    ],
    validation_data=valid,
    verbose=2
)

model.load_weights(f'{model_type}/')
model.save(f'{model_type}/model.h5')
