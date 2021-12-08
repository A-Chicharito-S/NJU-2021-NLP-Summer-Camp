from keras.models import load_model, Model
import numpy as np
import keras
import settings
import re
sub_name = '_restaurant_vocab5000_300'
test_x = np.load('data_processed/test_data' + sub_name + '.npy')
test_y = np.load('data_processed/test_label' + sub_name + '.npy')

test_y = keras.utils.to_categorical(test_y, settings.CATEGORY)
model = load_model('IAN_restaurant_vocab5000_300_rms.hd5')
test_x_Context = test_x[:, :78, :]
test_x_Target = test_x[:, 78:, :]

T_att_model = Model(inputs=model.input, outputs=model.get_layer('make_att').output)
print('predicting...')
_, T_att = T_att_model.predict([test_x_Context, test_x_Target])
T_att = T_att.squeeze()

C_att_model = Model(inputs=model.input, outputs=model.get_layer('make_att_1').output)
print('predicting...')
_, C_att = C_att_model.predict([test_x_Context, test_x_Target])
C_att = C_att.squeeze()

with open(settings.RESTAURANTS_TEST_WORDS, 'r', encoding='UTF=8') as r:
    with open(settings.RESTAURANTS_TEST_PROCESSED, 'w', encoding='UTF-8') as w:
        lines = r.readlines()
        i = 0
        for line in lines:
            i = i + 1
            index = i % 2
            if index == 0:  # context
                line = re.sub('/1[0-9]+', '', line)
                line = re.sub('/1', ' _', line, count=1)
                index = line.split().index('_')
                if index == 1:
                    line = re.sub(' _', '', line, count=1)
                    line = '_ '+line
                    index = 0
                line = re.sub('/[0-9]*', '', line)
                w.write(line)
                att_c = C_att[int(i/2)-1:int(i/2), :len(line.strip().split())-1].reshape(-1)
                # print('i{},len{},shape{}'.format(i,len(line.strip().split())-1, att_c.shape))
                att_c = np.insert(att_c, index, 0, axis=0)
                att_c = att_c/att_c.sum()  # normalize only for the actual words in the context att
                content = str(att_c.tolist())
                # content = re.sub(']|\[', '', content)
                # content = re.sub(',', ' ', content)
                w.write(content + '\n')  # att for the context
            else:           # target
                w.write(line)
                att_t = T_att[int((i+1)/2)-1:int((i+1)/2), :len(line.strip().split())].squeeze()
                att_t = att_t/att_t.sum()  # normalize only for the actual words in the target att
                content = str(att_t.tolist())
                # content = re.sub(']|\[', '', content)
                # content = re.sub(',', ' ', content)
                w.write(content + '\n')  # att for the target
    w.close()
r.close()
