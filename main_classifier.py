import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
from sklearn.metrics import confusion_matrix

# Loading the training data
PATH = os.getcwd()
data_path = 'train_img'  # Modify data_path with your train image set path
data_dir_list = os.listdir(data_path)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = x / 255
        img_data_list.append(x)

img_data = np.array(img_data_list)
print(img_data.shape)
img_data = np.rollaxis(img_data, 1, 0)
print(img_data.shape)
img_data = img_data[0]
print(img_data.shape)

# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')
print(labels.shape)

step = 1400  # number of AF training img
step1 = 1600  # number of N training img
step2 = 1240  # number of ST training img
step3 = 1600  # number of VF training img
labels[step * 0:step - 1] = 0
labels[step:step + step1 - 1] = 1
labels[step + step1:step + step1 + step2 - 1] = 2
labels[step + step1 + step2:step + step1 + step2 + step3 - 1] = 3

names = ['AF', 'Normal', 'ST', 'VF']

Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

#########################################################################################
image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
model.summary()
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-1]:
    layer.trainable = False

custom_vgg_model.layers[2].trainable
# conf. weight name
checkpoint = ModelCheckpoint(filepath='b30e200.hdf5',
                             monitor='loss',
                             mode='min',
                             save_best_only=True)

custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

t = time.time()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=30, epochs=200, verbose=1, validation_data=(X_test, y_test),
                            callbacks=[checkpoint])
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=30, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

####################################################################################################################
# %%
# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(200)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.title(' ')
plt.grid(True)
plt.legend(['train', 'validation'])
plt.plot()

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('number of epochs')
plt.ylabel('accuracy')
plt.title(' ')
plt.grid(True)
plt.legend(['train', 'validation'], loc=4)
plt.plot()


# %%


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


new_model = keras.models.load_model('b30e200.hdf5')
new_model.summary()

new_model.compile(optimizer=new_model.optimizer,  
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

test_dir = os.path.join('test_img')   # Modify data_path with your test image set path
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=1, target_size=(224, 224), color_mode='rgb',
                                                  shuffle=False)
test_generator.reset()

output = new_model.predict_generator(test_generator, steps=1460)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)

predicted_class_indices = np.argmax(output, axis=1)
labels = (test_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
print(labels)
predictions = [labels[k] for k in predicted_class_indices]

# filenames=test_generator.filenames[test_generator.index_array]
filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})

print(results)

class_names = ['af', 'normal', 'st', 'vf']  # Alphanumeric order1

print('-- Confusion Matrix --')
test_generator.reset()
Y_pred = new_model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

plot_confusion_matrix(confusion_matrix(test_generator.classes[test_generator.index_array], y_pred), normalize=True,
                      target_names=class_names, title='Confusion matrix, without normalization')
plot_confusion_matrix(confusion_matrix(test_generator.classes[test_generator.index_array], y_pred), normalize=False,
                      target_names=class_names, title='Confusion matrix, without normalization')
plt.show()
