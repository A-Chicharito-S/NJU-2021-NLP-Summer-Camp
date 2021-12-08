by changing different configs in settings.py:
<br> <br>
lap+rest: total number of words: 7883,   total training sentence: 7694 ,   vocab is set to 7000/7500,   word_embedding_size is set to 200/300
<br>
lap: total number of words: 4183,   total training sentence: 2966,   vocab is set to 4000,   word_embedding_size is set to 200
<br>
rest: total number of words: 5270,   total training sentence: 4728,   vocab is set to 5000,   word_embedding_size is set to 200
<br> <br>
and then run dataset.py, you will create different (data_xxx.npy, label_xxx.npy) files in this directory
<br><br> by running test_laptop.py and test_rest.py you can create laptops_text_processed.txt and restaurants_text_processed.txt where targets/contexts and their attention weights are stored
