from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, Xception, NASNetMobile
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import numpy as np
from scipy import ndimage
from numpy.random import seed
from tensorflow import random as random_
import pickle
import tensorflow as tf

seed_ = 1337                # сид для модуля рандома
seed(seed_)
random_.set_seed(seed_)

def add_noise(img):             # просто функция добавление шума по Гауссу
    rad = random.choice([1, 3, 3])
    img = ndimage.gaussian_filter(img, rad)
    deviation = variability_random*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    img += variability * random.gauss(0, 1)
    np.clip(img, 0., 255.)
    return img

variability = 5                 # величина изменчивости шума
variability_random = 5          # ^^^^^^^^
batch_size = 16                 # количество изображений, которое видеокарта обрабатывает за раз(параллельно, чем выше объём, тем больше доступно)
n_epochs = 25                   # количество эпох
lr = 1e-3                       # learning rate - скорость обучения
# аугментация базы данных:
datagen = ImageDataGenerator(
    preprocessing_function=
    add_noise,                  # добавление нашего шума
    horizontal_flip=True,       # отзеркаливание
    fill_mode="mirror",         # режим заполнения данных за границами доступного набора - в данном случае режимом отзеркаливания  dcba|abcd|dcba, где abcd - доступный кусочек
    rotation_range=5,           # дипазон поворота для случайных вращений в градусах
    rescale=1. / 255,           # превращение значений пикселов (0..255) в число (от 0 до 1)

    # далее всё в частях от изначальной информации:
    shear_range=0.1,            # диапазон обрезки
    zoom_range=0.1,             # диапазон приближения/отдаления
    width_shift_range=0.1,      # диапазон сдвига по x
    height_shift_range=0.1,     # диапазон сдвига по y
    validation_split=0.1,       # какая часть пойдет в валидационный сет
    )
last_layers = 7                 # количество последних слоев в готовых моделях, которые будут заменены на собственные обучаемые слои

gpus = tf.config.experimental.list_physical_devices('GPU') # все доступные физические GPU
if gpus: # если они есть, попробовать включить на каждом увеличение памяти
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


def f1(y_true, y_pred):     # метрика F1
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def soft_f1(y, y_hat):     # метрика F1 
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1
    macro_cost = tf.reduce_mean(cost)
    return macro_cost


def make_net(base):
    base = base(input_shape=img_shape + (3,), include_top=False, weights="imagenet")
    for layer in base.layers[:-last_layers]:
        layer.trainable = False
    head = base.output
    # head = MaxPooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(64, activation='relu')(head)
    head = Dropout(0.5)(head)
    head = Dense(1, activation='sigmoid')(head)
    model = Model(inputs=base.input, outputs=head)
    return model


if __name__ == '__main__':
    for Base, img_shape in [(NASNetMobile, (224, 224)), (ResNet50V2, (331, 331)), (ResNet101V2, (331, 331)), (Xception, (331, 331))]:
        model = make_net(Base)
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=['accuracy', f1])
        #K.set_value(model.optimizer.learning_rate, lr)

        train_generator = datagen.flow_from_directory(
            'dataset',  # this is the target directory
            target_size=img_shape,  # all images will be resized
            batch_size=batch_size,
            class_mode="binary",
            subset="training")  # since we use binary_crossentropy loss, we need binary labels

        val_generator = datagen.flow_from_directory(
            'dataset',  # this is the target directory
            target_size=img_shape,  # all images will be resized
            batch_size=batch_size,
            class_mode='binary',
            subset="validation")  # since we use binary_crossentropy loss, we need binary labels

        es = EarlyStopping(monitor="val_f1", mode='max', patience=5, restore_best_weights=True)
        mcp = ModelCheckpoint(f'models/model-{Base.__name__}-' '{epoch:03d}.h5', monitor='val_f1', mode='max')
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=n_epochs,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            callbacks=[es, mcp])
        model.save(f'models/model-{Base.__name__}.h5')  # always save your weights after training or during training
        pickle.dump(history.history, open(f"stories/history-{Base.__name__}.pkl", 'wb'))

