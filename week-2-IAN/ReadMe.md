IAN
====
This is the project for week 2, using an architecture called Interactive Attetion Network to do aspective-level sentiment classification.<br>
Using the Restaurants and Laptops datasets---which can be downloaded via https://github.com/songyouwei/ABSA-PyTorch (or simply use the dataset under Zhaofei/data)<br>
Using pre-trained word vector Glove---6B version from https://nlp.stanford.edu/projects/glove/
<br>The project consists mainly of four parts:
1. **dataset.py**---**reads sentences in the dataset and converts them to** 
-----
              1.a NumPy array whose elements are:
                [sentence1, sentence2,...], where words in each sentence are replaced with corresponding word embeddings.
                sentences are also padded with zero vectors to have the same length.
                shape: (total sentences in dataset x (max_context_length + max_target_length) x word embedding size)
                
              2.a NumPy array whose elements are:
                [0 or 1 or 2, 0 or 1 or 2,...] where if the snetence is positive then 2, neutral then 1, negative then 0
                shape: a 1-d vector with length of total sentences in the dataset
                
2. **model.py**---**reads the max_S_len (for context), max_T_len (for target) and returns a model with:**
------
                2 parellel LSTM layers, one for encoding context, another for encoding target
                2 parellel self-defined attention layer to calculate the attention of context/target with respect to the other
                
                a concatenate layer to concatenate the outputs from the attention layer
                
                a flatten layer to flatten the output from concatenation layer before dense layers
                
                a dense layer of 3 neurons whose activation function is 'tanh'
                a softmax layer to convert the output from the previous dense layer to probabilities
                
3. **train.py**---**train and evaluate the model**
 ------
               using categorical_crossentropy loss, with a train/test split of (0.8:0.2)
               using early stopping with a min_delta of 0.001 and patience of 50
               
4. **settings.py**---**stores the configurations of the model**
 ------
**experiment details**:<br>
===
We tried two configurations as follows:<br>
1. dataset: laptops, vocabulary size: 4000 (out of 4203), word embedding size: 300
2. dataset: restaurants, vocabulary size: 5000 (out of 5276), word embedding size: 300

**Test acc/loss**:
1. for laptops:
-----
Starting training.<br>
Epoch 1/50
73/73 [==============================] - 95s 1s/step - loss: 1.1029 - acc: 0.4149 - val_loss: 1.0542 - val_acc: 0.5405<br>
Epoch 2/50
73/73 [==============================] - 94s 1s/step - loss: 1.0228 - acc: 0.4858 - val_loss: 0.9775 - val_acc: 0.5732<br>
Epoch 3/50
73/73 [==============================] - 101s 1s/step - loss: 0.9789 - acc: 0.5228 - val_loss: 0.9246 - val_acc: 0.5919<br>
Epoch 4/50
73/73 [==============================] - 102s 1s/step - loss: 0.9455 - acc: 0.5477 - val_loss: 0.8852 - val_acc: 0.6168<br>
Epoch 5/50
73/73 [==============================] - 66s 898ms/step - loss: 0.9352 - acc: 0.5597 - val_loss: 0.9269 - val_acc: 0.6137<br>
Epoch 6/50
73/73 [==============================] - 102s 1s/step - loss: 0.9453 - acc: 0.5503 - val_loss: 0.8801 - val_acc: 0.6402<br>
Epoch 7/50
73/73 [==============================] - 65s 886ms/step - loss: 0.8935 - acc: 0.5743 - val_loss: 0.8598 - val_acc: 0.6308<br>
Epoch 8/50
73/73 [==============================] - 65s 890ms/step - loss: 0.8737 - acc: 0.5833 - val_loss: 0.8810 - val_acc: 0.6168<br>
Epoch 9/50
73/73 [==============================] - 65s 895ms/step - loss: 0.8985 - acc: 0.5692 - val_loss: 0.9072 - val_acc: 0.6277<br>
Epoch 10/50
73/73 [==============================] - 62s 855ms/step - loss: 0.8870 - acc: 0.5769 - val_loss: 0.9164 - val_acc: 0.5857<br>
Epoch 11/50
73/73 [==============================] - 112s 2s/step - loss: 0.8641 - acc: 0.5881 - val_loss: 0.8380 - val_acc: 0.6604<br>
Epoch 12/50
73/73 [==============================] - 66s 911ms/step - loss: 0.8836 - acc: 0.5726 - val_loss: 0.8626 - val_acc: 0.6371<br>
Epoch 13/50
73/73 [==============================] - 67s 924ms/step - loss: 0.8664 - acc: 0.5936 - val_loss: 0.9068 - val_acc: 0.6184<br>
Epoch 14/50
73/73 [==============================] - 75s 1s/step - loss: 0.8563 - acc: 0.5966 - val_loss: 0.8484 - val_acc: 0.6464<br>
Epoch 15/50
73/73 [==============================] - 66s 909ms/step - loss: 0.8273 - acc: 0.5984 - val_loss: 0.8271 - val_acc: 0.6464<br>
Epoch 16/50
73/73 [==============================] - 68s 926ms/step - loss: 0.8342 - acc: 0.6125 - val_loss: 0.9035 - val_acc: 0.6231<br>
Epoch 17/50
73/73 [==============================] - 67s 913ms/step - loss: 0.8068 - acc: 0.6237 - val_loss: 0.8851 - val_acc: 0.6277<br>
Epoch 18/50
73/73 [==============================] - 67s 911ms/step - loss: 0.8483 - acc: 0.6104 - val_loss: 0.9814 - val_acc: 0.5545<br>
Epoch 19/50
73/73 [==============================] - 65s 894ms/step - loss: 0.8092 - acc: 0.6241 - val_loss: 0.8252 - val_acc: 0.6526<br>
Epoch 20/50
73/73 [==============================] - 67s 921ms/step - loss: 0.8008 - acc: 0.6409 - val_loss: 0.8495 - val_acc: 0.6371<br>
Epoch 21/50
73/73 [==============================] - 112s 2s/step - loss: 0.7894 - acc: 0.6430 - val_loss: 0.8228 - val_acc: 0.6729<br>
Epoch 22/50
73/73 [==============================] - 61s 841ms/step - loss: 0.7654 - acc: 0.6598 - val_loss: 0.8284 - val_acc: 0.6386<br>
Epoch 23/50
73/73 [==============================] - 61s 831ms/step - loss: 0.7982 - acc: 0.6233 - val_loss: 0.8792 - val_acc: 0.6153<br>
Epoch 24/50
73/73 [==============================] - 58s 794ms/step - loss: 0.8349 - acc: 0.6078 - val_loss: 0.8360 - val_acc: 0.6620<br>
Epoch 25/50
73/73 [==============================] - 58s 795ms/step - loss: 0.7708 - acc: 0.6495 - val_loss: 0.8924 - val_acc: 0.6090<br>
Epoch 26/50
73/73 [==============================] - 55s 748ms/step - loss: 0.7481 - acc: 0.6534 - val_loss: 0.8285 - val_acc: 0.6558<br>
Epoch 27/50
73/73 [==============================] - 55s 749ms/step - loss: 0.7425 - acc: 0.6602 - val_loss: 0.8711 - val_acc: 0.6168<br>
Epoch 28/50
73/73 [==============================] - 55s 748ms/step - loss: 0.7358 - acc: 0.6761 - val_loss: 0.8629 - val_acc: 0.6417<br>
Epoch 29/50
73/73 [==============================] - 55s 757ms/step - loss: 0.7397 - acc: 0.6632 - val_loss: 0.9003 - val_acc: 0.6199<br>
Epoch 30/50
73/73 [==============================] - 65s 892ms/step - loss: 0.7405 - acc: 0.6675 - val_loss: 0.8418 - val_acc: 0.6573<br>
Epoch 31/50
73/73 [==============================] - 60s 817ms/step - loss: 0.7104 - acc: 0.6834 - val_loss: 0.8831 - val_acc: 0.6386<br>
Epoch 32/50
73/73 [==============================] - 60s 820ms/step - loss: 0.6943 - acc: 0.6916 - val_loss: 0.8192 - val_acc: 0.6682<br>
Epoch 33/50
73/73 [==============================] - 62s 843ms/step - loss: 0.7134 - acc: 0.6830 - val_loss: 0.8804 - val_acc: 0.6449<br>
Epoch 34/50
73/73 [==============================] - 60s 821ms/step - loss: 0.7036 - acc: 0.6830 - val_loss: 0.8833 - val_acc: 0.6402<br>
Epoch 35/50
73/73 [==============================] - 62s 846ms/step - loss: 0.6968 - acc: 0.6886 - val_loss: 0.8539 - val_acc: 0.6495<br>
Epoch 36/50
73/73 [==============================] - 62s 846ms/step - loss: 0.6881 - acc: 0.6942 - val_loss: 0.8488 - val_acc: 0.6558<br>
Epoch 37/50
73/73 [==============================] - 58s 798ms/step - loss: 0.6726 - acc: 0.7070 - val_loss: 0.8665 - val_acc: 0.6371<br>
Epoch 38/50
73/73 [==============================] - 60s 818ms/step - loss: 0.6593 - acc: 0.7040 - val_loss: 0.9454 - val_acc: 0.6184<br>
Epoch 39/50
73/73 [==============================] - 61s 830ms/step - loss: 0.6584 - acc: 0.6980 - val_loss: 0.9614 - val_acc: 0.5981<br>
Epoch 40/50
73/73 [==============================] - 62s 844ms/step - loss: 0.6918 - acc: 0.6761 - val_loss: 0.8301 - val_acc: 0.6464<br>
Epoch 41/50
73/73 [==============================] - 61s 837ms/step - loss: 0.6665 - acc: 0.7066 - val_loss: 0.8815 - val_acc: 0.6308<br>
Epoch 42/50
73/73 [==============================] - 60s 818ms/step - loss: 0.6624 - acc: 0.6976 - val_loss: 0.8444 - val_acc: 0.6589<br>
Epoch 43/50
73/73 [==============================] - 60s 825ms/step - loss: 0.6437 - acc: 0.7118 - val_loss: 0.9211 - val_acc: 0.6246<br>
Epoch 44/50
73/73 [==============================] - 60s 823ms/step - loss: 0.6586 - acc: 0.6899 - val_loss: 0.9026 - val_acc: 0.6340<br>
Epoch 45/50
73/73 [==============================] - 60s 824ms/step - loss: 0.6390 - acc: 0.7062 - val_loss: 0.8354 - val_acc: 0.6620<br>
Epoch 46/50
73/73 [==============================] - 61s 840ms/step - loss: 0.6287 - acc: 0.7174 - val_loss: 0.8714 - val_acc: 0.6573<br>
Epoch 47/50
73/73 [==============================] - 60s 816ms/step - loss: 0.6516 - acc: 0.7135 - val_loss: 0.8899 - val_acc: 0.6417<br>
Epoch 48/50
73/73 [==============================] - 60s 824ms/step - loss: 0.6598 - acc: 0.7143 - val_loss: 0.8907 - val_acc: 0.6324<br>
Epoch 49/50
73/73 [==============================] - 59s 810ms/step - loss: 0.6485 - acc: 0.7027 - val_loss: 0.9565 - val_acc: 0.5888<br>
Epoch 50/50
73/73 [==============================] - 55s 753ms/step - loss: 0.6812 - acc: 0.6976 - val_loss: 0.8983 - val_acc: 0.6277<br>
Done. Now evaluating.
21/21 [==============================] - 4s 203ms/step - loss: 0.8228 - acc: 0.6729<br>
Test accuracy: 0.67, loss: 0.82<br>


2. for restaurants:
-----
Starting training.<br>
Epoch 1/50
116/116 [==============================] - 129s 1s/step - loss: 1.0385 - acc: 0.4439 - val_loss: 0.9010 - val_acc: 0.6446<br>
Epoch 2/50
116/116 [==============================] - 124s 1s/step - loss: 1.0151 - acc: 0.4563 - val_loss: 0.8688 - val_acc: 0.6675<br>
Epoch 3/50
116/116 [==============================] - 135s 1s/step - loss: 0.9576 - acc: 0.4915 - val_loss: 0.8273 - val_acc: 0.7090<br>
Epoch 4/50
116/116 [==============================] - 92s 795ms/step - loss: 0.9313 - acc: 0.4928 - val_loss: 0.7831 - val_acc: 0.6975<br>
Epoch 5/50
116/116 [==============================] - 93s 801ms/step - loss: 0.9463 - acc: 0.5012 - val_loss: 0.8192 - val_acc: 0.7090<br>
Epoch 6/50
116/116 [==============================] - 92s 794ms/step - loss: 0.9184 - acc: 0.5107 - val_loss: 0.7864 - val_acc: 0.6993<br>
Epoch 7/50
116/116 [==============================] - 92s 795ms/step - loss: 0.9131 - acc: 0.5093 - val_loss: 0.7722 - val_acc: 0.6958<br>
Epoch 8/50
116/116 [==============================] - 92s 795ms/step - loss: 0.8940 - acc: 0.5212 - val_loss: 0.8038 - val_acc: 0.6702<br>
Epoch 9/50
116/116 [==============================] - 137s 1s/step - loss: 0.8897 - acc: 0.5242 - val_loss: 0.7375 - val_acc: 0.7407<br>
Epoch 10/50
116/116 [==============================] - 106s 914ms/step - loss: 0.8954 - acc: 0.5223 - val_loss: 0.7549 - val_acc: 0.7240<br>
Epoch 11/50
116/116 [==============================] - 106s 914ms/step - loss: 0.8707 - acc: 0.5399 - val_loss: 0.7279 - val_acc: 0.7399<br>
Epoch 12/50
116/116 [==============================] - 106s 915ms/step - loss: 0.8755 - acc: 0.5401 - val_loss: 0.7399 - val_acc: 0.7222<br>
Epoch 13/50
116/116 [==============================] - 106s 915ms/step - loss: 0.8488 - acc: 0.5531 - val_loss: 0.7418 - val_acc: 0.7257<br>
Epoch 14/50
116/116 [==============================] - 148s 1s/step - loss: 0.8486 - acc: 0.5550 - val_loss: 0.7186 - val_acc: 0.7434<br>
Epoch 15/50
116/116 [==============================] - 101s 872ms/step - loss: 0.8567 - acc: 0.5374 - val_loss: 0.7266 - val_acc: 0.7319<br>
Epoch 16/50
116/116 [==============================] - 145s 1s/step - loss: 0.8281 - acc: 0.5539 - val_loss: 0.7203 - val_acc: 0.7522<br>
Epoch 17/50
116/116 [==============================] - 102s 880ms/step - loss: 0.8326 - acc: 0.5620 - val_loss: 0.7532 - val_acc: 0.7240<br>
Epoch 18/50
116/116 [==============================] - 102s 881ms/step - loss: 0.8164 - acc: 0.5715 - val_loss: 0.7211 - val_acc: 0.7346<br>
Epoch 19/50
116/116 [==============================] - 102s 881ms/step - loss: 0.8358 - acc: 0.5610 - val_loss: 0.7233 - val_acc: 0.7434<br>
Epoch 20/50
116/116 [==============================] - 102s 881ms/step - loss: 0.8206 - acc: 0.5642 - val_loss: 0.7062 - val_acc: 0.7496<br>
Epoch 21/50
116/116 [==============================] - 102s 879ms/step - loss: 0.8084 - acc: 0.5758 - val_loss: 0.7073 - val_acc: 0.7496<br>
Epoch 22/50
116/116 [==============================] - 102s 881ms/step - loss: 0.8167 - acc: 0.5780 - val_loss: 0.7851 - val_acc: 0.7019<br>
Epoch 23/50
116/116 [==============================] - 102s 879ms/step - loss: 0.8326 - acc: 0.5658 - val_loss: 0.7191 - val_acc: 0.7407<br>
Epoch 24/50
116/116 [==============================] - 102s 881ms/step - loss: 0.8063 - acc: 0.5777 - val_loss: 0.7269 - val_acc: 0.7407<br>
Epoch 25/50
116/116 [==============================] - 102s 880ms/step - loss: 0.7835 - acc: 0.5880 - val_loss: 0.7036 - val_acc: 0.7399<br>
Epoch 26/50
116/116 [==============================] - 102s 882ms/step - loss: 0.7809 - acc: 0.5856 - val_loss: 0.7003 - val_acc: 0.7496<br>
Epoch 27/50
116/116 [==============================] - 102s 879ms/step - loss: 0.7761 - acc: 0.5896 - val_loss: 0.7088 - val_acc: 0.7443<br>
Epoch 28/50
116/116 [==============================] - 145s 1s/step - loss: 0.7858 - acc: 0.5753 - val_loss: 0.6912 - val_acc: 0.7593<br>
Epoch 29/50
116/116 [==============================] - 101s 868ms/step - loss: 0.7713 - acc: 0.5831 - val_loss: 0.7121 - val_acc: 0.7416<br>
Epoch 30/50
116/116 [==============================] - 101s 872ms/step - loss: 0.7550 - acc: 0.6058 - val_loss: 0.7097 - val_acc: 0.7425<br>
Epoch 31/50
116/116 [==============================] - 101s 875ms/step - loss: 0.7409 - acc: 0.6096 - val_loss: 0.6954 - val_acc: 0.7593<br>
Epoch 32/50
116/116 [==============================] - 101s 874ms/step - loss: 0.7790 - acc: 0.5853 - val_loss: 0.7002 - val_acc: 0.7566<br>
Epoch 33/50
116/116 [==============================] - 101s 874ms/step - loss: 0.7446 - acc: 0.6139 - val_loss: 0.6961 - val_acc: 0.7540<br>
Epoch 34/50
116/116 [==============================] - 102s 876ms/step - loss: 0.7536 - acc: 0.5950 - val_loss: 0.7035 - val_acc: 0.7531<br>
Epoch 35/50
116/116 [==============================] - 102s 876ms/step - loss: 0.7405 - acc: 0.5994 - val_loss: 0.7287 - val_acc: 0.7337<br>
Epoch 36/50
116/116 [==============================] - 102s 876ms/step - loss: 0.7365 - acc: 0.6096 - val_loss: 0.7075 - val_acc: 0.7549<br>
Epoch 37/50
116/116 [==============================] - 102s 877ms/step - loss: 0.7460 - acc: 0.6010 - val_loss: 0.7532 - val_acc: 0.7196<br>
Epoch 38/50
116/116 [==============================] - 102s 879ms/step - loss: 0.7145 - acc: 0.6204 - val_loss: 0.7006 - val_acc: 0.7584<br>
Epoch 39/50
116/116 [==============================] - 102s 878ms/step - loss: 0.7283 - acc: 0.6131 - val_loss: 0.7067 - val_acc: 0.7478<br>
Epoch 40/50
116/116 [==============================] - 102s 877ms/step - loss: 0.7226 - acc: 0.6156 - val_loss: 0.7024 - val_acc: 0.7451<br>
Epoch 41/50
116/116 [==============================] - 102s 878ms/step - loss: 0.7088 - acc: 0.6191 - val_loss: 0.6799 - val_acc: 0.7593<br>
Epoch 42/50
116/116 [==============================] - 146s 1s/step - loss: 0.7058 - acc: 0.6242 - val_loss: 0.6971 - val_acc: 0.7637<br>
Epoch 43/50
116/116 [==============================] - 97s 832ms/step - loss: 0.7163 - acc: 0.6188 - val_loss: 0.6968 - val_acc: 0.7504<br>
Epoch 44/50
116/116 [==============================] - 97s 836ms/step - loss: 0.7038 - acc: 0.6237 - val_loss: 0.7306 - val_acc: 0.7222<br>
Epoch 45/50
116/116 [==============================] - 141s 1s/step - loss: 0.6897 - acc: 0.6426 - val_loss: 0.6852 - val_acc: 0.7646<br>
Epoch 46/50
116/116 [==============================] - 103s 891ms/step - loss: 0.6888 - acc: 0.6386 - val_loss: 0.7001 - val_acc: 0.7496<br>
Epoch 47/50
116/116 [==============================] - 103s 891ms/step - loss: 0.6800 - acc: 0.6402 - val_loss: 0.7075 - val_acc: 0.7513<br>
Epoch 48/50
116/116 [==============================] - 103s 891ms/step - loss: 0.6836 - acc: 0.6334 - val_loss: 0.7288 - val_acc: 0.7399<br>
Epoch 49/50
116/116 [==============================] - 103s 891ms/step - loss: 0.7122 - acc: 0.6272 - val_loss: 0.7172 - val_acc: 0.7416<br>
Epoch 50/50
116/116 [==============================] - 103s 891ms/step - loss: 0.6811 - acc: 0.6302 - val_loss: 0.7077 - val_acc: 0.7593<br>
Done. Now evaluating.
36/36 [==============================] - 8s 228ms/step - loss: 0.6843 - acc: 0.7663<br>
Test accuracy: 0.77, loss: 0.68<br>

**graph for train/validation acc/loss**:
1. for laptops<br>
![image](https://user-images.githubusercontent.com/61720358/127845094-f241d5cc-3ec9-4226-9e61-f516e3d3e1f9.png)

2. for restaurants<br>
![image](https://user-images.githubusercontent.com/61720358/127845146-31b7ed07-8d84-457b-ad77-dceea92b3134.png)


**attention visualization**:<br>
1. for laptops<br>
label = 2, prediction = 2<br>
![image](https://user-images.githubusercontent.com/61720358/127855195-89cf672f-680e-4ca6-bf1b-ac4c206941ab.png)<br>
label = 2, prediction = 2<br>
![image](https://user-images.githubusercontent.com/61720358/127855240-ee8fc091-d369-4eeb-9adf-129078cb4fb8.png)<br>
label = 2, prediction = 2<br>
![image](https://user-images.githubusercontent.com/61720358/127855268-3daa2a21-1c72-438e-9df1-a34acdafe6ee.png)<br>
label = 0, prediction = 0<br>
![image](https://user-images.githubusercontent.com/61720358/127855284-85ff1943-b7da-4939-a9ef-aacd9378a3b2.png)<br>

2. for restaurants<br>
label = 0, prediction = 0<br>
![image](https://user-images.githubusercontent.com/61720358/127845244-7537d9f1-fbc8-45cb-ac45-9bf7d26c88ae.png)<br>
label = 2, prediction = 2<br>
![image](https://user-images.githubusercontent.com/61720358/127845285-6144c849-ebca-4db7-8bb0-017fa9ef2419.png)<br>
label = 2, prediction = 2<br>
![image](https://user-images.githubusercontent.com/61720358/127845306-bc5072a8-22fe-4af4-8708-d871e2a69919.png)<br>


<br><br>
**Note**：
<br>We find that the attention results for both laptops and restaurants datasets are not that good.
<br>For sentences with small length, the attention weights tend to be the average for each words.
<br>For sentences with large length, the attentions weights tend to be the average within a certain window round the target words
<br>We also find that under some circumstances the attention weigths for a emotion word (i.e. excellent), is even lower than other non-emotional words











