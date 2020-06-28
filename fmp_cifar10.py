#!/usr/bin/env python3
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Lambda, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler as lr_scheduler
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Dense


def summary(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	plt.legend(loc=0)
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	plt.legend(loc=0)
	# save plot to file
	plt.subplots_adjust(wspace=None, hspace=1.0)
	filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
	plt.savefig(filename + ".png")
	plt.close()

def lr_decay(epoch):
	lr = 0.001
	if epoch > 75:
		lr = 0.0005
	if epoch > 125:
		lr = 0.0002
	if epoch > 350:
		lr = 0.0001
	return lr

def frac_max_pool(x):
    return tf.nn.fractional_max_pool(x, [1.0, 1.41, 1.41, 1.0], pseudo_random=True, overlapping=True)[0]

def mdlTrain(train_feature, train_label, test_feature, test_label):
	# model definition
	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	#model.add(Lambda(frac_max_pool))  # frac_max_pool
	model.add(Dropout(0.3))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Lambda(frac_max_pool))  # frac_max_pool
	model.add(Dropout(0.35))

	model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Lambda(frac_max_pool))  # frac_max_pool
	model.add(Dropout(0.35))

	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Lambda(frac_max_pool))  # frac_max_pool
	model.add(Dropout(0.4))

	model.add(Conv2D(filters=160, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Conv2D(filters=160, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Lambda(frac_max_pool))  # frac_max_pool
	model.add(Dropout(0.45))

	model.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(Lambda(frac_max_pool))  # frac_max_pool
	model.add(Dropout(0.5))

	model.add(Conv2D(filters=192, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform'))
	model.add(LeakyReLU())
	model.add(BatchNormalization())
	model.add(GlobalAveragePooling2D())
	model.add(Dense(units=10, kernel_initializer='he_uniform', activation='softmax'))

	print(model.summary())
	
	# training definition
	batch_num = 64
	epoch_num = 600
	opt = RMSprop(decay=1e-6)
	datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	datagen.fit(train_feature)
	checkpoint = ModelCheckpoint("cifar10_best.h5", monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
	csv_logger = CSVLogger('training.csv')
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	history = model.fit_generator(datagen.flow(train_feature, train_label, batch_size=batch_num), \
		steps_per_epoch=int(len(train_feature) / batch_num), epochs=epoch_num, validation_data=(test_feature, test_label), \
		verbose=2, callbacks=[lr_scheduler(lr_decay), checkpoint, csv_logger])	
	summary(history)

	# accuracy evaluation
	accuracy = model.evaluate(test_feature, test_label)
	print('\n[Accuracy] = ', accuracy[1])

	return model

# ---------------------------------------------------------------------------

# load cifar10 data
(train_feature, train_label), (test_feature, test_label) = cifar10.load_data()

# data preprocessing
# reshape
train_feature_vector = train_feature.reshape(len(train_feature), 32, 32, 3).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 32, 32, 3).astype('float32')

# feature normalization
# z-score
mean = np.mean(train_feature_vector, axis=(0, 1, 2, 3))
std = np.std(train_feature_vector, axis=(0, 1, 2, 3))
train_feature_normal = (train_feature_vector - mean) / (std + 1e-7)
test_feature_normal = (test_feature_vector - mean) / (std + 1e-7)

# one-hot encoding
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

# train model
model = mdlTrain(train_feature_normal, train_label_onehot, test_feature_normal, test_label_onehot)
accuracy = model.evaluate(test_feature_normal, test_label_onehot)
print('\n[Accuracy] = ', accuracy)

# save model
filename = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
model.save(filename + ".h5")
del model