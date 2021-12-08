import tensorflow as tf
import settings
import Dataset
import model
import keras
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt

word_list, word_frequency, word_dict, maxlen = Dataset.create_vocabulary()
word_embedding_dict = Dataset.create_word_embedding_dict(word_dict)
data, label = Dataset.make_dataset(word_embedding_dict, maxlen)
label = to_categorical(label, 2)
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)

train_x = tf.expand_dims(train_x, -1)
test_x = tf.expand_dims(test_x, -1)

model = model.construct_model(maxlen)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=settings.LR,
                                                          decay_steps=settings.DECAY_STEP, decay_rate=settings.LR_DECAY)
sgd = SGD(learning_rate=lr_schedule, momentum=settings.MOMENTUM)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])
savemodel = ModelCheckpoint(settings.CHECKPOINT_PATH, save_best_only=True)
stopmodel = EarlyStopping(min_delta=settings.MIN_DELTA, patience=settings.PATIENCE)

history = model.fit(x=train_x, y=train_y, batch_size=settings.BATCH_SIZE, epochs=settings.EPOCH,
                    verbose=1, validation_data=(test_x, test_y), shuffle=True, callbacks=[savemodel,stopmodel])


model = load_model(settings.CHECKPOINT_PATH)
loss, acc = model.evaluate(x=test_x, y=test_y)
print("Test accuracy: %3.2f, loss: %3.2f"%(acc, loss))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
