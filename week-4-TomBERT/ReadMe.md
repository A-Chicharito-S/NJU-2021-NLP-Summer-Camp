TomBERT
======
This is the week 4 project---TomBERT<br><br><br>
## The structure of the program is:<br>
![image](https://user-images.githubusercontent.com/61720358/128211928-2a5c529c-c4e0-47df-a56b-c40d0c1afc48.png)
<br><br>
## The structure of *train.py*:<br>
![image](https://user-images.githubusercontent.com/61720358/128491711-b4444442-1a20-4a83-814a-2cab92cd9f4d.png)
<br><br>
## The structure of *test.py*:<br>
![image](https://user-images.githubusercontent.com/61720358/128491480-4e2349f2-1e0d-448c-adaf-60061163d9d9.png)
<br><br>
## The structure of *model.py*:<br> 
![image](https://user-images.githubusercontent.com/61720358/128511185-7d561ed4-b547-41c5-8dc7-a2bdf4f04fd3.png)
<br><br>
## The strucuture of **TrainData** class of *data_helper.py*:<br><br>
### init():<br>
**vacab_path**: path of bert self-defined vocabularies to serve as a tool for word-->number mapping<br>
**output_path**: path of files that store label-->index mapping (e.g.: positive-->2, or 1-->2)<br>
**picture_path**: path of files that store the image feature<br>
**sequence_len**: max sequence len (used for padding all sentences to the same length, in paper = 64)<br>
**target_len**: max sequence len (used for padding all targets to the same length, in paper = 16)<br>
**batch_size**: size of every batch<br>
<br>
### read_data():<br>
returns:<br>
**inputs**: a list, stores the inputs (context + target) of one bert model<br>
**targets**: a list, stores the targets of another bert model<br>
**pictures**: a list, stores the picture indexes (e.g.: 167321.jpg)<br>
**labels**: a list, stores the labels of each training piece (e.g.: [0, -1, 1, 1, ...])<br>
<br>
### text_cleaner():<br>
clean the text data (e.g.: remove multi-spaces etc.)<br>
<br>
### trans_to_index():<br>
transform word-->number (e.g.: I like apple-->0 4 3), only for targets (e.g.: [CLS] + sentence + [SEP])<br>
**input_id**: the numeric sequence of one input sentence<br>
**input_ids**: the set of all 'input_id's<br>
**input_masks**: a set contains all lists which are all '1' lists with length of corresponding 'input_id'<br>
**segment_ids**: a set contains all lists which are all '0' lists with length of corresponding 'input_id'<br>
<br>
### \_truncated_seg_pair():<br>
first judge the two sentences (context + target) > max input length for bert (= 512)<br>
if true, use list.pop() to truncate two sentences to the max input length<br>
called in trans_to_index_input()<br>
<br>
### trans_to_index_input():<br>
transform word-->number (e.g.: I like apple-->0 4 3), only for targets (e.g.: [CLS] + sentence A + [SEP] + sentence B + [SEP])<br>
**input_id**: the numeric sequence of the two input sentences<br>
**input_ids**: the set of all 'input_id's<br>
**input_masks**: a set contains all lists which are all '1' lists with length of corresponding 'input_id'<br>
**segment_ids**: a set contains all lists which are all '0' and '1' lists (e.g.: all '0' of len([CLS] + sentence A + [SEP]), all '1' of len(sentence B + [SEP])) with length of corresponding 'input_id'<br>
<br>
### picture_feature():<br>
return a list whose elements are numpy arrays which are representations of corresponidng picture of the 'picture_id's<br>
<br>
### trans_label_to_index():<br>
with help of the label-->numeric index mapping stored in self.output_path, translate labels to numeric indexes<br>
<br>
### padding():<br>
for targets (with form of [CLS] + sentence + [SEP]) and texts (with form of [CLS] + sentence A + [SEP] + sentence B + [SEP])<br>
pad the targets to target's max length, and adjust the input_masks (increase length or truncate) and segment_ids (increase length or truncate)<br>
padding_position():<br>
not useful in this scenario since position_id isn't specified in raw data<br>
<br>
### gen_data():<br>
prepare the data, get:<br>
input_ids, input_masks, segment_ids for texts<br>
input_ids, input_masks, segment_ids for targets<br>
picture_id, label_ids, label_to_index mapping and return them<br>
<br>
### next_batch():<br>
load the dataset batch by batch<br>
one batch include corresponding:<br>
input_ids, input_masks, segment_ids for texts in this batch<br>
input_ids, input_masks, segment_ids for targets in this batch<br>
picture_id, label_ids in this batch<br>
<br><br>
## The strucuture of **Predictor** class of *predict.py*:<br><br>
### load_graph():<br>
load the compute graph from the save checkpoint<br>
<br>
### create_model():<br>
create a model using BertClassifier from *model.py*<br>
<br>
### the rest methods:<br>
basically with the same logic and functionality of what they are in *data_helper.py*<br>
<br><br>
## result for laptop (replacing LSTM with BERT in IAN):<br>
### train:<br>
![image](https://user-images.githubusercontent.com/61720358/129133717-6e27a7bb-6d80-4c6c-bf29-61cd84b3deeb.png)
### test:<br>
![image](https://user-images.githubusercontent.com/61720358/129133776-6648bafc-a40b-47fe-ac2a-d6acb605b590.png)
## result for restaurant (replacing LSTM with BERT in IAN):<br>
### train:<br>
![image](https://user-images.githubusercontent.com/61720358/129133862-4a6aa7fa-06a3-42cc-a8e3-1e57c9fdf0d5.png)
### test:<br>
![image](https://user-images.githubusercontent.com/61720358/129133909-1e9a3ad6-d435-4563-b6e1-5cff3854cbb3.png)
