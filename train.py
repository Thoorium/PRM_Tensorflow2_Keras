import tensorflow as tf

from model import PeakResponseMapping

import os
import sys
import json

num_classes = 1
backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(448,448,3), pooling=None, classes=num_classes)
#backbone.trainable=False
model = tf.keras.Sequential()
model.add(backbone)
model.add(PeakResponseMapping(filter_type='mean'))
model.add(tf.keras.layers.Dense(1))

#model.build((1,448,448,3))
#model.summary()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(image, label):
    with tf.GradientTape() as tape:
        predictions = model(image)
        loss = loss_object(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, predictions)


@tf.function
def test_step(image, label):
    predictions = model(image)
    t_loss = loss_object(label, predictions)

    test_loss(t_loss)
    test_accuracy(label, predictions)


EPOCHS = 5
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [448, 448])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def load_dataset(images_path: str, labels_path: str):
    images = []
    labels = []

    annotations = json.load(open(labels_path))
    annotations = list(annotations.values())
    annotations = [a for a in annotations if a['regions']]
    for a in annotations:
            label_group = []
            for aa in a['regions']:
                if "seg" not in aa['region_attributes']:
                    continue
                if aa['region_attributes']['seg'] == "":
                    continue

                ##label_group.append(aa['region_attributes']['seg'])
                ##if aa['region_attributes']['seg'] == 'pt':
                ##    label_group.append(1)
                ##if aa['region_attributes']['seg'] == 'tf':
                ##    label_group.append(2)
            #labels.append(label_group)
            labels.append('pt')

            image_path = os.path.join(images_path, a['filename'].split('/')[-1])
            if not os.path.isfile(image_path):
                continue

            
            images.append(image_path)
         
    return images, labels

image_paths, labels = load_dataset('/Users/olivierbeaudoin/Projects/PRM_Tensorflow2_Keras/_dataset/images', '/Users/olivierbeaudoin/Projects/PRM_Tensorflow2_Keras/_dataset/labels.json')

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(labels)
train_seqs = tokenizer.texts_to_sequences(labels)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_seqs, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=300))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

ds_train = ds.take(270)
ds_test = ds.skip(270).take(30)

for epoch in range(EPOCHS):
    for image, label in ds_train:
        train_step(image, label)

    for test_image, test_label in ds_test:
        test_step(test_image, test_label)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
