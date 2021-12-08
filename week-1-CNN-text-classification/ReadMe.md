CNN-Text-Classification
====
This is the project for week 1, using a CNN-architecture for text classification.<br>
Using the Cornell Movie Review dataset---sentence polarity dataset v1.0 from https://www.cs.cornell.edu/people/pabo/movie-review-data/<br>
Using pre-trained word vector Glove---6B version from https://nlp.stanford.edu/projects/glove/
<br>The project consists of four parts:
1. **dataset.py**---**reads sentences in the dataset and converts them to** 
-----
              1.a NumPy array whose elements are:
                [sentence1, sentence2,...], where words in each sentence are replaced with corresponding word embeddings.
                sentences are also padded with zero vectors to have the same length.
                shape: (total sentences in dataset x max_sentence_length x word embedding size)
                
              2.a NumPy array whose elements are:
                [0 or 1, 0 or 1,...] where if the snetence is positive then 1, negative then 0
                shape: a 1-d vector with length of total sentences in the dataset
                
2. **model.py**---**reads the max_sentence_length and returns a model with:**
------
                3 parellel CNN layers, with kernel size of (3/4/5) x word embedding size, 100 channels each
                3 corresponding max pooling layer to extract the max value for each channel with respective kernels
                3 reshape layer to reshape each kernel's features to 1 x 100 to achieve concatenation
                
                a concatenate layer to concatenate 1 x 100 tensors from different kernels to a 1 x 300 one
                
                a flatten layer to flatten the output from concatenation layer before dense layers
                
                a dense layer of 1024 neurons
                a dense layer of 2 neurons
                
3. **train_eval.py**---**train and evaluate the model**
 ------
               using binary_crossentropy loss, with a train/test split of (0.8:0.2)
               using early stopping with a min_delta of 0.001 and patience of 50
               
4. **settings.py**---**stores the configurations of the model**
 ------
**Note**:
<br>  To run the code, you need to download the pre-trained word embedding GloVe on your own and create a file called 'check' of name extension '.hdf5' and store them under the same directory with the codes and dataset provided here.
<br>
<br>
**Train/Validation acc/loss**:
![image](https://user-images.githubusercontent.com/61720358/126040576-da5b5a20-8084-45ee-8abc-e734894983af.png)
<br><br>
**The final test acc is 0.77 and loss is around 0.49**
