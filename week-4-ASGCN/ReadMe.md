ASGCN-DT
===
this is another week 4 project using GCN-based model to do sentiment classification<br>
The original paper is: **Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks** [[paper]](https://arxiv.org/abs/1909.03477)<br>
source of paper: https://arxiv.org/pdf/1909.03477.pdf<br>
only model.py is uploaded since time is limited<br>
and only the semEval 2014 laptops dataset is used for experiment
## Training process for Laptops:<br>
Starting training.<br>
Epoch 1/50<br>
73/73 [==============================] - 104s 1s/step - loss: 1.0216 - acc: 0.4867 - val_loss: 1.0030 - val_acc: 0.5794<br>
Epoch 2/50<br>
73/73 [==============================] - 103s 1s/step - loss: 0.9336 - acc: 0.5567 - val_loss: 0.8994 - val_acc: 0.5857<br>
Epoch 3/50<br>
73/73 [==============================] - 195s 3s/step - loss: 0.8948 - acc: 0.5679 - val_loss: 0.8681 - val_acc: 0.6231<br>
Epoch 4/50<br>
73/73 [==============================] - 205s 3s/step - loss: 0.8498 - acc: 0.6001 - val_loss: 0.8289 - val_acc: 0.6604<br>
Epoch 5/50<br>
73/73 [==============================] - 176s 2s/step - loss: 0.8376 - acc: 0.5936 - val_loss: 0.8316 - val_acc: 0.6386<br>
Epoch 6/50<br>
73/73 [==============================] - 215s 3s/step - loss: 0.8140 - acc: 0.6078 - val_loss: 0.8267 - val_acc: 0.6729<br>
Epoch 7/50<br>
73/73 [==============================] - 166s 2s/step - loss: 0.7867 - acc: 0.6426 - val_loss: 0.7934 - val_acc: 0.6713<br>
Epoch 8/50<br>
73/73 [==============================] - 214s 3s/step - loss: 0.7681 - acc: 0.6353 - val_loss: 0.8239 - val_acc: 0.6807<br>
Epoch 9/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.7355 - acc: 0.6542 - val_loss: 0.7959 - val_acc: 0.6760<br>
Epoch 10/50<br>
73/73 [==============================] - 76s 1s/step - loss: 0.7227 - acc: 0.6585 - val_loss: 0.7879 - val_acc: 0.6807<br>
Epoch 11/50<br>
73/73 [==============================] - 115s 2s/step - loss: 0.6903 - acc: 0.6761 - val_loss: 0.8237 - val_acc: 0.6838<br>
Epoch 12/50<br>
73/73 [==============================] - 83s 1s/step - loss: 0.6990 - acc: 0.6778 - val_loss: 0.8452 - val_acc: 0.6526<br>
Epoch 13/50<br>
73/73 [==============================] - 122s 2s/step - loss: 0.6623 - acc: 0.6912 - val_loss: 0.7848 - val_acc: 0.6869<br>
Epoch 14/50<br>
73/73 [==============================] - 73s 1s/step - loss: 0.6270 - acc: 0.7092 - val_loss: 0.8120 - val_acc: 0.6776<br>
Epoch 15/50<br>
73/73 [==============================] - 73s 1s/step - loss: 0.6351 - acc: 0.7191 - val_loss: 0.8269 - val_acc: 0.6667<br>
Epoch 16/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.6419 - acc: 0.7006 - val_loss: 0.8347 - val_acc: 0.6838<br>
Epoch 17/50<br>
73/73 [==============================] - 71s 970ms/step - loss: 0.6248 - acc: 0.7152 - val_loss: 0.8438 - val_acc: 0.6667<br>
Epoch 18/50<br>
73/73 [==============================] - 69s 950ms/step - loss: 0.6231 - acc: 0.7088 - val_loss: 0.8297 - val_acc: 0.6620<br>
Epoch 19/50<br>
73/73 [==============================] - 67s 914ms/step - loss: 0.6234 - acc: 0.7088 - val_loss: 0.8288 - val_acc: 0.6729<br>
Epoch 20/50<br>
73/73 [==============================] - 66s 908ms/step - loss: 0.6027 - acc: 0.7182 - val_loss: 0.8296 - val_acc: 0.6854<br>
Epoch 21/50<br>
73/73 [==============================] - 67s 912ms/step - loss: 0.6015 - acc: 0.7332 - val_loss: 0.8514 - val_acc: 0.6745<br>
Epoch 22/50<br>
73/73 [==============================] - 97s 1s/step - loss: 0.5759 - acc: 0.7363 - val_loss: 0.8167 - val_acc: 0.6916<br>
Epoch 23/50<br>
73/73 [==============================] - 71s 973ms/step - loss: 0.5868 - acc: 0.7345 - val_loss: 0.8518 - val_acc: 0.6667<br>
Epoch 24/50<br>
73/73 [==============================] - 71s 974ms/step - loss: 0.6075 - acc: 0.7156 - val_loss: 0.8351 - val_acc: 0.6791<br>
Epoch 25/50<br>
73/73 [==============================] - 71s 977ms/step - loss: 0.5906 - acc: 0.7302 - val_loss: 0.8649 - val_acc: 0.6589<br>
Epoch 26/50<br>
73/73 [==============================] - 71s 969ms/step - loss: 0.5862 - acc: 0.7393 - val_loss: 0.8611 - val_acc: 0.6682<br>
Epoch 27/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5632 - acc: 0.7427 - val_loss: 0.8896 - val_acc: 0.6495<br>
Epoch 28/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5543 - acc: 0.7479 - val_loss: 0.8559 - val_acc: 0.6807<br>
Epoch 29/50<br>
73/73 [==============================] - 73s 1s/step - loss: 0.5816 - acc: 0.7414 - val_loss: 0.8479 - val_acc: 0.6745<br>
Epoch 30/50<br>
73/73 [==============================] - 78s 1s/step - loss: 0.5681 - acc: 0.7427 - val_loss: 0.8499 - val_acc: 0.6776<br>
Epoch 31/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5317 - acc: 0.7676 - val_loss: 0.8505 - val_acc: 0.6885<br>
Epoch 32/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5329 - acc: 0.7552 - val_loss: 0.8442 - val_acc: 0.6729<br>
Epoch 33/50<br>
73/73 [==============================] - 80s 1s/step - loss: 0.5383 - acc: 0.7509 - val_loss: 0.8412 - val_acc: 0.6745<br>
Epoch 34/50<br>
73/73 [==============================] - 79s 1s/step - loss: 0.5376 - acc: 0.7607 - val_loss: 0.8987 - val_acc: 0.6449<br>
Epoch 35/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5365 - acc: 0.7564 - val_loss: 0.8723 - val_acc: 0.6651<br>
Epoch 36/50<br>
73/73 [==============================] - 88s 1s/step - loss: 0.5307 - acc: 0.7573 - val_loss: 0.8599 - val_acc: 0.6729<br>
Epoch 37/50<br>
73/73 [==============================] - 78s 1s/step - loss: 0.5168 - acc: 0.7655 - val_loss: 0.8981 - val_acc: 0.6417<br>
Epoch 38/50<br>
73/73 [==============================] - 88s 1s/step - loss: 0.5231 - acc: 0.7556 - val_loss: 0.8844 - val_acc: 0.6589<br>
Epoch 39/50<br>
73/73 [==============================] - 95s 1s/step - loss: 0.5207 - acc: 0.7689 - val_loss: 0.8620 - val_acc: 0.6651<br>
Epoch 40/50<br>
73/73 [==============================] - 79s 1s/step - loss: 0.5324 - acc: 0.7642 - val_loss: 0.8777 - val_acc: 0.6495<br>
Epoch 41/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5524 - acc: 0.7470 - val_loss: 0.8758 - val_acc: 0.6589<br>
Epoch 42/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5555 - acc: 0.7453 - val_loss: 0.8964 - val_acc: 0.6589<br>
Epoch 43/50<br>
73/73 [==============================] - 83s 1s/step - loss: 0.5284 - acc: 0.7560 - val_loss: 0.8826 - val_acc: 0.6620<br>
Epoch 44/50<br>
73/73 [==============================] - 83s 1s/step - loss: 0.5028 - acc: 0.7715 - val_loss: 0.8937 - val_acc: 0.6604<br>
Epoch 45/50<br>
73/73 [==============================] - 79s 1s/step - loss: 0.5040 - acc: 0.7801 - val_loss: 0.8969 - val_acc: 0.6417<br>
Epoch 46/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5282 - acc: 0.7620 - val_loss: 0.8797 - val_acc: 0.6713<br>
Epoch 47/50<br>
73/73 [==============================] - 78s 1s/step - loss: 0.5381 - acc: 0.7440 - val_loss: 0.8620 - val_acc: 0.6651<br>
Epoch 48/50<br>
73/73 [==============================] - 79s 1s/step - loss: 0.5381 - acc: 0.7547 - val_loss: 0.9075 - val_acc: 0.6542<br>
Epoch 49/50<br>
73/73 [==============================] - 77s 1s/step - loss: 0.5356 - acc: 0.7603 - val_loss: 0.8481 - val_acc: 0.6698<br>
Epoch 50/50<br>
73/73 [==============================] - 79s 1s/step - loss: 0.5170 - acc: 0.7693 - val_loss: 0.8750 - val_acc: 0.6651<br>
Done. Now evaluating.<br>
21/21 [==============================] - 4s 210ms/step - loss: 0.8167 - acc: 0.6916<br>
Test accuracy: 0.69, loss: 0.82<br>
## Training/validation Acc/Loss for Laptops:<br>
![image](https://user-images.githubusercontent.com/61720358/129339603-a55f7a2f-1518-4bf7-b593-366a1fca2c6d.png)

