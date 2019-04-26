import tensorflow as tf

from model import fc_resnet50, PeakResponseMapping

backbone = fc_resnet50(num_classes=2, pretrained=True)
model = PeakResponseMapping(backbone)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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

def load_dataset(path: str, labels: str):
    return ''#wip

for epoch in range(EPOCHS):
  for image, label in mnist_train:
    train_step(image, label)

  for test_image, test_label in mnist_test:
    test_step(test_image, test_label)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))