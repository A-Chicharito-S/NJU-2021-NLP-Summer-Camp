import keras
import model
import dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop
import settings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

control_dict = {'laptops': 1, 'restaurant': 0}
_, _, _, max_S_len, max_T_len = dataset.create_vocabulary(laptop=control_dict['laptops'],
                                                          restaurant=control_dict['restaurant'])
if control_dict['laptops'] == 1 and control_dict['restaurant'] == 0:
    sub_name = '_laptop_vocab4000_300'
if control_dict['laptops'] == 0 and control_dict['restaurant'] == 1:
    sub_name = '_restaurant_vocab5000_300'

train_x = np.load('data_processed/train_data' + sub_name + '.npy')
train_y = np.load('data_processed/train_label' + sub_name + '.npy')
test_x = np.load('data_processed/test_data' + sub_name + '.npy')
test_y = np.load('data_processed/test_label' + sub_name + '.npy')

train_y = keras.utils.to_categorical(train_y, settings.CATEGORY)
test_y = keras.utils.to_categorical(test_y, settings.CATEGORY)

MODEL_NAME = 'IAN' + sub_name + '.hd5'

train_x_Context = train_x[:, :max_S_len, :]
train_x_Target = train_x[:, max_S_len:, :]
test_x_Context = test_x[:, :max_S_len, :]
test_x_Target = test_x[:, max_S_len:, :]
print('train_x_Context shape: {}'.format(train_x_Context.shape))
print('train_x_Target shape: {}'.format(train_x_Target.shape))
print('test_x_Context shape: {}'.format(test_x_Context.shape))
print('test_x_Target shape: {}'.format(test_x_Target.shape))
print('max_S_len:{}'.format(max_S_len))
print('max_T_len:{}'.format(max_T_len))

model = model.make_model(max_S_len, max_T_len)
model.compile(optimizer=SGD(lr=settings.LR, momentum=settings.MOMENTUM), loss='categorical_crossentropy',
              metrics=['acc'])

# optimizer=tf.compat.v1.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
model.summary()

savemodel = ModelCheckpoint(filepath=MODEL_NAME, monitor='val_acc', save_best_only=False)
# stopmodel = EarlyStopping(min_delta=settings.MIN_DELTA, patience=settings.PATIENCE)
print("Starting training.")

history = model.fit(x=[train_x_Context, train_x_Target], y=train_y, batch_size=settings.BATCH_SIZE,
                    validation_data=([test_x_Context, test_x_Target], test_y),
                    shuffle=True, epochs=settings.EPOCH, callbacks=[savemodel])
# callbacks=[savemodel, stopmodel]

print("Done. Now evaluating.")
loss, acc = model.evaluate(x=[test_x_Context, test_x_Target], y=test_y)
print("Test accuracy: %3.2f, loss: %3.2f" % (acc, loss))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(100, 50))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training Accuracy-laptop') ######### remember to change here and below!!!!
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss-laptop')
plt.title('Training Loss')
plt.legend()
plt.show()
