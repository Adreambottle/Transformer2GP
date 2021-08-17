#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import os
import pandas as pd
import logging
import sklearn
from torch.utils.data import Dataset, DataLoader
import time
import random
from transformers import BertModel, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    trainer_callback, DataCollatorForTokenClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import gc
import sys
import re
from seqeval.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[2]:

def insert_deliminter(item):
    """
    inserts deliminters to description
    用于将item中的list类型的description添加特殊分隔符，转换成str
    输入是一个item，输出也是一个item
    """
    deliminter_list = [',', '/', ';', ':', '-', '_']  # '#' is removed
    deliminter = random.sample(deliminter_list, 1)[0]
    des = item['description']
    for i in range(len(des)):
        if des[i] != '':
            if random.uniform(0, 1) > 0.3:  ## new deliminter insertion
                des[i] += deliminter
            else:
                des[i] += random.sample(deliminter_list, 1)[0]
    item['description'] = re.sub('(' + '|'.join(deliminter_list) + ')$', '', ''.join(des))
    return item

def get_input_from_file_single(path, data_num, text_list, label_list, class_list, file_num, train_flag):
    """
    暂时不用，用于读取单个的文件
    这个函数是用来从 Cap 和 Res 里面读取原始数据，生成一个分割好的 data
    每次调用的时候都需要重新读取一遍数据
    path:        现在是: "preprocess/standard_cap.json"
                        "preprocess/standard_res.json"
    data_num:    是数据的个数，每个文件里面有多少条数据
    file_num:    是生成几个拆分好的文件包
    train_flag:  是用于train还是test
    class_list:  是什么
    text_list:   把生成的sample的'description'添加到text_list
    label_list:  把生成的sample的 class label 添加到label_list
    """
    with open(path, 'r', encoding='utf-8') as f:

        # standard_data 是内存需要读取的文件
        standard_data = json.loads(f.read())
        # print('sys.getsizeof(standard_data)',sys.getsizeof(standard_data))
        # sample_data = random.sample(standard_data,343*data_num)

    print('\tsys.getsizeof(standard_data)', sys.getsizeof(standard_data))
    # 这个是standard_data的数据大小

    standard_data_length = len(standard_data)
    print('\tlen of standard_data:', standard_data_length)
    # 这个是standard_data的数据长度

    # 不同category的匹配
    execute_num = 1
    # execute_num = 338

    if train_flag:
        # 如果是train的话，将数据进行拆分，生成一个sample_data
        lower_bound = file_num * data_num * execute_num
        upper_bound = min(standard_data_length, (file_num + 1) * data_num * execute_num)
        sample_data = standard_data[lower_bound:upper_bound]

    else:
        # 如果是test的话，只选取最前面的部分
        upper_bound = min(standard_data_length, data_num * execute_num)
        sample_data = standard_data[0:upper_bound]

    # 在内存读取 sample_data 之后，直接将原始数据删除
    del standard_data
    gc.collect()

    for item in sample_data:
        # 将 sample_data 中的 class，label 和 text 部分分别添加到列表里面
        # item = sample_data[0]

        # 生成 value 和 key 的空列表
        tmp_val = []
        tmp_key = []
        shuffled_item = list(item.items())
        random.shuffle(shuffled_item)

        for key, val in shuffled_item:
            if key == 'class':
                class_list.append(val)
            else:
                tmp_val.append(str(val))
                tmp_key.append(key)
        text_list.append(tmp_val)
        label_list.append(tmp_key)

    return text_list, label_list, class_list


def get_input_from_file(path, data_num, text_list, label_list, class_list, file_num, train_flag):
    """
    用于读取一个路径下的全部文件
    这个函数是用来从 Cap 和 Res 里面读取原始数据，生成一个分割好的 data
    每次调用的时候都需要重新读取一遍数据
    path:        现在是: "./MultiData/cap"
                        "./MultiData/res"
    data_num:    是数据的个数，每个文件里面有多少条数据
    file_num:    是生成几个拆分好的文件包
    train_flag:  是用于train还是test
    class_list:  把生成的sample的 class label 添加到class_list
    text_list:   把生成的sample的'description'添加到text_list
    label_list:  把生成的sample的 label label 添加到label_list
    """
    files = os.listdir(path)
    files_json = []
    for file in files:
        if file[-5:] == ".json":
            files_json.append(file)

    files_json_num = 0
    for file in files_json:

        # file = files_json[0]
        # print(f"\t读取第{files_json_num}个原始数据")
        files_json_num += 1

        with open(path + '/' + file, 'r', encoding='utf-8') as f:

            # standard_data 是内存需要读取的文件
            standard_data = json.loads(f.read())
            # print('sys.getsizeof(standard_data)',sys.getsizeof(standard_data))
            # sample_data = random.sample(standard_data,343*data_num)

        # print('\tsys.getsizeof(standard_data)', sys.getsizeof(standard_data))
        # 这个是standard_data的数据大小

        standard_data_length = len(standard_data)
        # print('\tlen of standard_data:', standard_data_length)
        # 这个是standard_data的数据长度


        execute_num = 1
        # 这个是比例

        if train_flag:
            # 如果是train的话，将数据进行拆分，生成一个sample_data
            lower_bound = file_num * data_num * execute_num
            upper_bound = min(standard_data_length, (file_num + 1) * data_num * execute_num)
            sample_data = standard_data[lower_bound:upper_bound]

        else:
            # 如果是test的话，只选取最前面的部分
            upper_bound = min(standard_data_length, data_num * execute_num)
            sample_data = standard_data[0:upper_bound]

        # 在内存读取 sample_data 之后，直接将原始数据删除
        del standard_data
        gc.collect()

        for item in sample_data:
            # 将 sample_data 中的 class，label 和 text 部分分别添加到列表里面
            # item = sample_data[0]

            # 生成 value 和 key 的空列表
            tmp_val = []
            tmp_key = []
            shuffled_item = list(item.items())
            random.shuffle(shuffled_item)

            for key, val in shuffled_item:
                if key == 'class':
                    class_list.append(val)
                else:
                    tmp_val.append(str(val))
                    tmp_key.append(key)
            text_list.append(tmp_val)
            label_list.append(tmp_key)

    return text_list, label_list, class_list


def get_input_from_sampling(input_size, train_flag, file_num):
    """
    input_size: the number of items to sample
    input_size:  是 train_size
    returns the input for simpletransformer
    """

    # 设置数据读取的路径，本地和服务器的路径是不一样的
    cap_path = '/data/bob/synthetic/cap/'
    res_path = '/data/bob/synthetic/res/'

    # cap_path = './MultiData/cap'
    # res_path = './MultiData/res'



    # 将抽取到的结果转化为输入格式
    text_list = []
    label_list = []
    class_list = []

    if train_flag:
        text_list, label_list, class_list = get_input_from_file(cap_path, input_size, text_list,
                                                                label_list, class_list, file_num,
                                                                train_flag)
        text_list, label_list, class_list = get_input_from_file(res_path, input_size, text_list,
                                                                label_list, class_list, file_num,
                                                                train_flag)
    else:
        text_list, label_list, class_list = get_input_from_file(cap_path, input_size,
                                                                text_list, label_list, class_list,
                                                                file_num, train_flag)
        text_list, label_list, class_list = get_input_from_file(res_path, input_size,
                                                                text_list, label_list, class_list, file_num,
                                                                train_flag)

    # to shuffle input
    res = []
    for i in range(len(label_list)):
        res.append((text_list[i], label_list[i], class_list[i]))
    random.shuffle(res)
    for i in range(len(label_list)):
        text_list[i] = res[i][0]
        label_list[i] = res[i][1]
        class_list[i] = res[i][2]

    print('query数量：', len(text_list))     # print 的是shuffled之后的数据长度
    print('label数量：', len(label_list))
    print('class数量：', len(class_list))

    #     print('sys.getsizeof(text_list)',sys.getsizeof(text_list))
    #     print('sys.getsizeof(label_list)',sys.getsizeof(label_list))

    return text_list, label_list, class_list


# a,b,c=get_input_from_sampling(500,True,1)
# for i in range(len(b)):
#     print((a[i],b[i],c[i]),'\n')


# In[3]:


# #保存列表，每行一个元素
# with open('input_example.txt','w',encoding='utf-8') as f:
#     d=[]
#     for i in range(len(b)):
#         d.append(c[i]+'       '+str(b[i])+'       '+str(a[i]))
#     f.write('\n'.join(d))


# In[4]:


def tokenize_and_align_labels(text, labels_before_split, tokenizer):
    tokenized_inputs = tokenizer(text, is_split_into_words=True, add_special_tokens=False,
                                 padding=True, truncation=False, return_tensors="pt")
    labels = []
    for i, label in enumerate(labels_before_split):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append('-100')
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    #    tokenized_inputs["labels"] = labels
    return tokenized_inputs, labels


def show_tokenizer_result(text, labels_before_split, tokenizer):
    tokenized_inputs, labels = tokenize_and_align_labels(text, labels_before_split, tokenizer)
    for i in range(len(labels_before_split)):
        print('text:', text[i])
        print('tokenized result:', tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][i]))
        # print('tokenized_inputs:',tokenized_inputs["input_ids"][i])

        print('labels_before_split:', labels_before_split[i])
        print('labels:', labels[i])
        print('\n%%%%%%%%%%%%%%%%%%%%%\n')


# show_tokenizer_result(a[:10],b[:10],tokenizer)


# In[5]:


# 读文件里面的数据转化为二维列表
def read_list(filename):
    """
    读取一个无txt后缀名的文件
    将"\t"作为分隔符，将每一行转为一个list
    返回的是一个 list_source
    """
    file1 = open(filename + ".txt", "r")
    list_row = file1.readlines()
    list_source = []
    for i in range(len(list_row)):
        column_list = list_row[i].strip().split("\t")  # 每一行split后是一个列表
        list_source.append(column_list)  # 在末尾追加到list_source
    file1.close()
    return list_source


# 保存二维列表到文件
def save_list(list1, filename):
    """
    把生成的嵌套list转换成一个txt文件
    """
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')  # 相当于Tab一下，换一个单元格
        file2.write('\n')      # 写完一行立马换行
    file2.close()


# In[6]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attrEncoder(all_class_list, item_class, attr):
    """
    就是编码
    """
    if item_class in all_class_list and attr in all_class_list[item_class]:
        if item_class == 'Capacitors':
            return all_class_list[item_class][attr]
        if item_class == 'Resistors':
            return all_class_list[item_class][attr] - 10
    if attr == '-100':
        return -100
    return 0


def attrDecoder(item_class, attr):
    if attr == 0 or attr == -100:
        return 'others'
    else:
        if item_class == 'Resistors':
            attr += 10
        global all_attrs_dict
        return all_attrs_dict[attr]


# attrDecoder('Resistors',3)


# In[7]:


# train_encodings = tokenizer(train_dataset, is_split_into_words=True,add_special_tokens=False,padding=True,truncation=True)
# for i in range(len(train_encodings)):
#     print('tokenized result:',tokenizer.convert_ids_to_tokens(train_encodings["input_ids"][i]))


# In[8]:


def train_input_process(train_size, tokenizer, file_num):
    """
    读取保存好的数据：【pt-encodings-text】【txt-lables-】
    train_size:     get_input_from_sampling 的第一个参数
    tokenizer:
    file_num
    """
    files = os.listdir('./inputData')
    # 打开这个路径下的全部文件

    #     with open('running_output.txt','w') as f:
    #         f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')

    # 准备训练数据
    print(f'train dataset{file_num}\n')
    if f'train_encodings{file_num}.pt' in files and f'train_labels{file_num}.txt' in files and f'train_class{file_num}.csv' in files:
        print('read train dataset\n')
        train_encodings = torch.load(f'./inputData/train_encodings{file_num}.pt')
        train_labels = read_list(f'./inputData/train_labels{file_num}')
        train_class = pd.read_csv(f'./inputData/train_class{file_num}.csv', index_col=0)
        train_class = train_class['0'].to_list()
    else:
        # 生成数据
        generate_start_time = time.time()
        print('generate train dataset\n')
        train_dataset, train_labels, train_class = get_input_from_sampling(train_size, train_flag=True,
                                                                           file_num=file_num)
        # train_encodings = tokenizer(train_dataset, is_split_into_words=True,add_special_tokens=False,
        #                        padding=True,truncation=False,return_tensors="pt")
        train_encodings, train_labels = tokenize_and_align_labels(train_dataset, train_labels, tokenizer)


        # 把 train_text 进行 encoding 编码， 保存成 pt 文件
        # 把 train_label 保存成没有后缀的 txt 文件
        # 把 train_class 保存成 csv 文件
        pd.DataFrame(train_class).to_csv(f'./inputData/train_class{file_num}.csv')
        save_list(train_labels, f'./inputData/train_labels{file_num}')
        torch.save(train_encodings, f'./inputData/train_encodings{file_num}.pt')
        generate_end_time = time.time()
        print(f'Train dataset saved, using time {generate_end_time - generate_start_time}s.')

    # encode the labels
    global all_class_list
    train_labels_encoded = []
    for i in range(len(train_labels)):
        #        train_labels_encoded[i] = list(map(lambda x,y,z:attrEncoder(x,y,z),all_class_list,train_class[i],train_labels[i]))
        train_labels_encoded.append(
            [attrEncoder(all_class_list, train_class[i], train_label) for train_label in train_labels[i]])

    print('training dataset is ok\n')
    return train_encodings, train_labels_encoded, train_class


def test_input_process(test_size, tokenizer):
    """
    """
    print('generate test dataset\n')
    generate_start_time = time.time()
    test_dataset, test_labels, test_class = get_input_from_sampling(test_size, train_flag=False, file_num=1)
    test_encodings, test_labels = tokenize_and_align_labels(test_dataset, test_labels, tokenizer)

    # encode the labels
    global all_class_list
    test_labels_encoded = []
    for i in range(len(test_labels)):
        test_labels_encoded.append(
            [attrEncoder(all_class_list, test_class[i], test_label) for test_label in test_labels[i]])

    print('testing dataset is ok\n')
    return test_encodings, test_labels_encoded, test_class, test_dataset[:1000], test_labels


# train_data,train_labels_encoded,train_class = train_input_process(model_checkpoint,3,tokenizer,1)
# te_data,te_labels_encoded,te_class,test_sample,test_labels = test_input_process(model_checkpoint,10,tokenizer)


# In[9]:


def ClassEncoder(Class):
    if Class == 'Capacitors':
        return 1
    elif Class == 'Resistors':
        return 2
    else:
        return 0


def ClassDecoder(Class):
    if Class == 1:
        return 'Capacitors'
    elif Class == 2:
        return 'Resistors'
    else:
        return 'Others'


# In[10]:


class processDataset(Dataset):
    def __init__(self, encodings, labels, classes):
        self.encodings = encodings
        self.labels = labels
        self.classes = classes

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['classes'] = torch.tensor(self.classes[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[11]:


class nerClass(torch.nn.Module):
    """
    定义ner的Class，是torch里面的一个层
    """
    def __init__(self, config):
        """
        config 是个全局变量     config = {'num_labels1': 9, 'num_labels2': 8, 'model': "prajjwal1/bert-tiny"}
                              config['num_labels1] = 9   分成9类
                              config['num_labels1] = 8   分成8类
                              config['model'] = "prajjwal1/bert-tiny" 预训练的模型
        """
        super(nerClass, self).__init__()
        self.num_labels1 = config['num_labels1']
        self.num_labels2 = config['num_labels2']
        self.l1 = BertModel.from_pretrained(config['model'])

        #         self.pre_classifier = torch.nn.Linear(128, 128)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier1 = torch.nn.Linear(768, config['num_labels1'])      # classifier1是一个线性分类器
        self.classifier2 = torch.nn.Linear(768, config['num_labels2'])

    #        self.init_weights()

    def forward(self, input_ids, attention_mask=None, targets=None, classes=None, device=None):
        #         output_1 = self.l1(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        #                             attention_mask=attention_mask, head_mask=head_mask)
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)

        """
        input_ids:
        attention_mask:
        targets:
        classes:
        device:
        """
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]   # 输出的第0个return
                                   # last_hidden_state (torch.FloatTensor of shape (
                                   #                    batch_size, sequence_length, hidden_size))
               # print('hidden_state.size()',hidden_state[:,0].size())
               # pooler = self.pre_classifier(pooler)
               # pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(hidden_state)
        #        print('pooler.size()',pooler.size())

        mask1 = torch.eq(classes, 1)    # classes 是输入，判断是否等于 1
        mask2 = torch.eq(classes, 2)
        #        print('pooler[mask1].size()',pooler[mask1].size())
        #        print('pooler[mask2].size()',pooler[mask2].size())

        # 这里是两个头，分开进行预测
        output1 = self.classifier1(pooler[mask1])  # 是pooler经过mask1筛选后的
        output2 = self.classifier2(pooler[mask2])  # 是pooler经过mask2筛选后的
        #         print('output1.size()',output1.size())
        #         print('output2.size()',output2.size())

        #        output1 = output1[mask1]
        target1 = targets[mask1]                  # targets是一个input的参数
        #         print('target1.size()',target1.size())

        #        output2 = output2[mask2]
        target2 = targets[mask2]
        #         print('target2.size()',target2.size())

        if output1.size()[0] != 0:
            if attention_mask[mask1] is not None:
                """
                # loss_function 是 CrossEntropy, 忽略 -100
                """
                active_loss1 = attention_mask[mask1].view(-1) == 1
                #            print('active_loss1',active_loss1.size())
                active_logits1 = output1.view(-1, config['num_labels1'])
                #                 print('active_logits1',active_logits1.size())
                active_labels1 = torch.where(
                    active_loss1, target1.view(-1), torch.tensor(loss_function.ignore_index).type_as(target1)
                )
                #                 print('active_labels1',active_labels1.size())
                loss1 = loss_function(active_logits1, active_labels1)
            else:
                loss1 = loss_function(output1.view(-1, config['num_labels1']), target1.view(-1))
        else:
            # 如果batch_size是0的话，loss1的值就是0
            loss1 = torch.tensor(0.0, requires_grad=True, device=device)

        if output2.size()[0] != 0:
            if attention_mask[mask2] is not None:
                active_loss2 = attention_mask[mask2].view(-1) == 1
                #            print('active_loss2',active_loss2.size())
                active_logits2 = output2.view(-1, config['num_labels2'])
                #                 print('active_logits2',active_logits2.size())
                active_labels2 = torch.where(
                    active_loss2, target2.view(-1), torch.tensor(loss_function.ignore_index).type_as(target2)
                )
                #                 print('active_labels2',active_labels2.size())
                loss2 = loss_function(active_logits2, active_labels2)
            else:
                loss2 = loss_function(output2.view(-1, config['num_labels2']), target2.view(-1))
        else:
            loss2 = torch.tensor(0.0, requires_grad=True, device=device)
        #         print(loss1)
        #        print(type(loss1))
        return output1, output2, mask1, mask2, loss1, loss2


# In[12]:


# convert subtokens to tokens
# for _,data in enumerate(testing_loader, start=0):
# #         print(_)
# #        print(data)
#         for item in data['input_ids']:
#             res=tokenizer.convert_ids_to_tokens(item)
#             #print(res)
#             res2=tokenizer.convert_tokens_to_string(res)
#             res2=res2.replace('[PAD]','')
# #            res2=res2.replace(' ',',')
#             print(res2)


# In[13]:


def train(model, epoch, cur_file, file_num, logging_step=1, training_loader=None, optimizer=None):
    tr_loss1 = 0
    tr_loss2 = 0
    tr_total_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    cnt = 0
    for _, data in enumerate(training_loader, start=0):
        """
        将training_loader::DataLoader里的数据用tensor的形式转到device上
        """
        ids = data['input_ids'].to(device, dtype=torch.long)
        attention_mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['labels'].to(device, dtype=torch.long)
        classes = data['classes'].to(device, dtype=torch.long)

        # 对梯度进行初始化,将梯度清零
        # 每一次train的时候都会清零梯度
        optimizer.zero_grad()

        # 前向传播
        output1, output2, mask1, mask2, loss1, loss2 = model(ids, attention_mask, targets, classes, device)

        # 保存每个epoch最后一个file的结果
        # if cur_file == file_num and cnt < 10:
        #
        #    cnt = cnt + 1
        #    if _ == 0:
        #        out_tensor1 = output1.clone()
        #        out_tensor2 = output2.clone()
        #        target_tensor1 = targets[mask1].clone()
        #        target_tensor2 = targets[mask2].clone()
        #    else:
        #        out_tensor1 = torch.cat((out_tensor1, output1), 0)
        #        out_tensor2 = torch.cat((out_tensor2, output2), 0)
        #        target_tensor1 = torch.cat((target_tensor1, targets[mask1]), 0)
        #        target_tensor2 = torch.cat((target_tensor2, targets[mask2]), 0)

        #print(loss1.item(), loss2.item())
       
        tr_loss1 += loss1.sum().item()
        tr_loss2 += loss2.sum().item()
        tr_total_loss += loss1.sum().item() + loss2.sum().item()

        # 反向传播
        # loss1.sum().backward(retain_graph=True)
        # loss2.sum().backward()
        loss = loss1.sum() + loss2.sum()
        loss.backward()
        # 对网络的参数进行更新
        optimizer.step()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        # logging
        if _ % logging_step == 0:
            loss_step = tr_total_loss / nb_tr_steps
            loss_step1 = tr_loss1 / nb_tr_steps
            loss_step2 = tr_loss2 / nb_tr_steps
            # accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss step{_}: total loss:{loss_step},cap loss:{loss_step1},res loss:{loss_step2}")
            # print(f"Training Accuracy per 5000 steps: {accu_step}")

    # # evaluate
    # if cur_file == file_num:
    #     print('evaluate training result\n')
    #     with open('class_classification_report.txt', 'a') as f:
    #         f.write(f'\n~~~~~~~~~~training epoch{epoch} result:~~~~~~~~~~\n')
    #     true_labels1, true_predictions1 = compute_metrics(True, target_tensor1.cpu().data.numpy().tolist(),
    #                                                       out_tensor1.argmax(-1).cpu().data.numpy().tolist(),
    #                                                       'Capacitors')
    #     true_labels2, true_predictions2 = compute_metrics(True, target_tensor2.cpu().data.numpy().tolist(),
    #                                                       out_tensor2.argmax(-1).cpu().data.numpy().tolist(),
    #                                                       'Resistors')

    # print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_total_loss / nb_tr_steps
    #    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    #    print(f"Training Accuracy Epoch: {epoch_accu}")
    return model


# model = train(1,1)


# In[14]:


def compute(predictions, references, suffix=False):
    report = classification_report(y_true=references, y_pred=predictions, suffix=suffix, output_dict=True)
    report.pop("macro avg")
    report.pop("weighted avg")
    overall_score = report.pop("micro avg")

    scores = {
        type_name: {
            "precision": score["precision"],
            "recall": score["recall"],
            "f1": score["f1-score"],
            "number": score["support"],
        }
        for type_name, score in report.items()
    }
    scores["overall_precision"] = overall_score["precision"]
    scores["overall_recall"] = overall_score["recall"]
    scores["overall_f1"] = overall_score["f1-score"]
    scores["overall_accuracy"] = accuracy_score(y_true=references, y_pred=predictions)

    return scores


# In[15]:


def compute_metrics(train_flag, labels, preds, classes):
    #    print(labels)
    # Remove ignored index (special tokens)
    global all_class_list
    true_predictions = [[attrDecoder(classes, p) for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(preds, labels)]
    true_labels = [[attrDecoder(classes, l) for (p, l) in zip(prediction, label) if l != -100]
                   for prediction, label in zip(preds, labels)]

    #     print(true_predictions)
    #     print(true_labels)
    results = compute(predictions=true_predictions, references=true_labels)
    # print(results)

    # 对result进行处理，以写入文件
    for k, v in results.items():
        if isinstance(v, dict):
            v['number'] = str(v['number'])

    global write_result
    print(f'write_result: {classes}')
    if write_result:
        #         file = open('class_classification_report.txt', 'w')
        #         for k,v in dict.items():
        #             file.write(k + ' ')
        #             for k1, v1 in v.items():
        #                 file.write(k1 + ' ' + str(v1) + ' ')
        #             file.write(' \n')
        #         file.close()
        #         class_preds = [attrDecoder(item) for item in true_predictions]
        #         class_labels = [attrDecoder(item) for item in true_labels]
        # class_report = classification_report(class_labels, class_preds,output_dict=True)
        # class_report = classification_report(class_labels, class_preds)
        #         with open('class_classification_report.txt','a') as f:
        #             f.write(str(results))

        with open('class_classification_report.json', 'a', errors='ignore') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        res = {"precision": results["overall_precision"],
               "recall": results["overall_recall"],
               "f1": results["overall_f1"],
               "accuracy": results["overall_accuracy"]}
        if train_flag:
            print('train_result:', res)
            with open('class_classification_report.txt', 'a') as f:
                f.write(str(res) + '\n')
        else:
            print('test_result:', res)
            with open('class_classification_report.txt', 'a') as f:
                f.write(str(res) + '\n')

    return true_labels, true_predictions


# In[22]:


def test(model, logging_step, testing_loader, optimizer, test_class_part, sample_data):
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss1 = 0
    tr_loss2 = 0
    tr_total_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        cnt = 0
        for _, data in enumerate(testing_loader, 0):

            ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            targets = data['labels'].to(device, dtype=torch.long)
            classes = data['classes'].to(device, dtype=torch.long)
            # print(classes.size())

            # 前向传播
            output1, output2, mask1, mask2, loss1, loss2 = model(ids, attention_mask, targets, classes, device)


            if cnt < 1:
                cnt = cnt + 1
                if _ == 0:
                    out_tensor1 = output1.clone()
                    out_tensor2 = output2.clone()
                    target_tensor1 = targets[mask1].clone()
                    target_tensor2 = targets[mask2].clone()
                else:
                    out_tensor1 = torch.cat((out_tensor1, output1), 0)
                    out_tensor2 = torch.cat((out_tensor2, output2), 0)
                    target_tensor1 = torch.cat((target_tensor1, targets[mask1]), 0)
                    target_tensor2 = torch.cat((target_tensor2, targets[mask2]), 0)

            #             for item in ids[mask1]:
            #                 res=tokenizer.convert_ids_to_tokens(item)
            #                 #print(res)
            #                 res=tokenizer.convert_tokens_to_string(res)
            #                 res=res.replace('[PAD]','')
            #                 input.append(res.strip())

            # print(loss1.item(), loss2.item())
            tr_loss1 += loss1.sum().item()
            tr_loss2 += loss2.sum().item()
            tr_total_loss += loss1.sum().item() + loss2.sum().item()

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # logging
            if _ % logging_step == 0:
                loss_step = tr_total_loss / nb_tr_steps
                loss_step1 = tr_loss1 / nb_tr_steps
                loss_step2 = tr_loss2 / nb_tr_steps
                # accu_step = (n_correct*100)/nb_tr_examples
                print(f"Testing Loss per step: total loss:{loss_step},cap loss:{loss_step1},res loss:{loss_step2}")
                # print(f"Training Accuracy per 5000 steps: {accu_step}")
    #     print(target_tensor1.size())
    #     print(out_tensor1.argmax(-1).size())

    with open('class_classification_report.txt', 'a') as f:
        f.write('\n~~~~~~~~~~testing result:~~~~~~~~~~\n')
    true_labels1, true_predictions1 = compute_metrics(False, target_tensor1.cpu().data.numpy().tolist(),
                                                      out_tensor1.argmax(-1).cpu().data.numpy().tolist(), 'Capacitors')
    true_labels2, true_predictions2 = compute_metrics(False, target_tensor2.cpu().data.numpy().tolist(),
                                                      out_tensor2.argmax(-1).cpu().data.numpy().tolist(), 'Resistors')

    cap_input = [v for i, v in zip(test_class_part, sample_data) if i == 'Capacitors']
    res_input = [v for i, v in zip(test_class_part, sample_data) if i == 'Resistors']

    min_len1 = min(len(true_labels1), len(cap_input))
    min_len2 = min(len(true_labels2), len(res_input))

    result_desc = cap_input[:min_len1] + res_input[:min_len2]
    result_labels = true_labels1[:min_len1] + true_labels2[:min_len2]
    result_predictions = true_predictions1[:min_len1] + true_predictions2[:min_len2]

    join_original_input = pd.DataFrame(
        {"description": result_desc, "labels": result_labels, 'predictions': result_predictions})
    join_original_input.to_csv('test_description_result.csv', encoding='utf_8_sig')

    #             if _%5000==0:
    #                 loss_step = tr_loss/nb_tr_steps
    #                 accu_step = (n_correct*100)/nb_tr_examples
    #                 print(f"Validation Loss per 100 steps: {loss_step}")
    #                 print(f"Validation Accuracy per 100 steps: {accu_step}")
    #     epoch_loss = tr_loss/nb_tr_steps
    #     epoch_accu = (n_correct*100)/nb_tr_examples
    #     print(f"Validation Loss Epoch: {epoch_loss}")
    #     print(f"Validation Accuracy Epoch: {epoch_accu}")

    return


# test(model,1)


# In[23]:


def main(num_epoch=10, save_epoch=2, logging_step=10, file_num=10, train_size=150, test_size=30):
    """
    logging_step:  每隔多少步打印一次结果
    train_size:    是train_input_process的第一个参数
    model: 就是一个 nerClass(config)
    """
    model = nerClass(config)  # model
    model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # train_data 和 test_data 的模型参数
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': NUM_WORKERS}

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': NUM_WORKERS}

    # training loop
    total_time_start = time.time()
    for j in range(num_epoch):
        print(f'\n\n%%%%%%%%%%%%%%epoch{j}%%%%%%%%%%%%%%\n\n')
        for i in range(file_num):
            print(f'\n\nepoch{j} begin generate train_dataset{i}\n\n')

            print('\nProcessing Training data\n')
            process_start_time = time.time()
            train_data, train_labels_encoded, train_class = train_input_process(train_size=train_size,
                                                                                tokenizer=tokenizer,
                                                                                file_num=i)
            # 在这一步添加了 ClassEncoder，将 class 转化成了 Encoder 的形式
            train_class_encoded = list(map(lambda x: ClassEncoder(x), train_class))

            # 打包成一个utils里的DataLoader的包
            train_dataset = processDataset(train_data, train_labels_encoded, train_class_encoded)
            training_loader = DataLoader(train_dataset, **train_params)
            process_end_time = time.time()
            print("processing time: ", process_end_time - process_start_time)




            # Train the model
            print('\nTraining Begins\n')
            train_start_time = time.time()
            model = train(model=model, epoch=j, cur_file=i + 1, file_num=file_num, logging_step=logging_step,
                          training_loader=training_loader, optimizer=optimizer)
            train_end_time = time.time()
            print(f"Training time: {train_end_time - train_start_time}")

            total_time_in_this_data = time.time()
            print(f"Training up to now using time: {total_time_in_this_data - total_time_start}")

            gc.collect()
            print("This sub data file has finished! \n\n")

        if j % save_epoch == 0:
            # 每隔 save_epoch 保存一次模型
            output_model_file = f'./models/model{j}/model.bin'
            output_vocab_file = f'./models/model{j}'
            if not os.path.exists(f'./models/model{j}'):
                os.makedirs(f'./models/model{j}')
            torch.save(model, output_model_file)
            tokenizer.save_pretrained(output_vocab_file)
            print('All files saved')

    total_time_end = time.time()
    print('\n\nTotal_training_time:', total_time_start - total_time_end)


    # generate test data
    # sample_data是测试集中前一千的数据
    test_data, test_labels_encoded, test_class, sample_data, test_labels = test_input_process(test_size=test_size,
                                                                                              tokenizer=tokenizer)
    test_class_encoded = list(map(lambda x: ClassEncoder(x), test_class))
    test_dataset = processDataset(test_data, test_labels_encoded, test_class_encoded)
    testing_loader = DataLoader(test_dataset, **test_params)

    # test process
    test(model=model, logging_step=logging_step, testing_loader=testing_loader, optimizer=optimizer,
         test_class_part=test_class[:1000], sample_data=sample_data)

    # 保存最终模型
    output_model_file = './models/model_final/model.bin'
    output_vocab_file = './models/model_final'
    if not os.path.exists('./models/model_final'):
        os.makedirs('./models/model_final')
    torch.save(model, output_model_file)
    tokenizer.save_pretrained(output_vocab_file)
    print('All files saved')


# In[24]:


if __name__ == "__main__":
    # 注意最开始的label处理部分，已经把没意义的赋值为-100了
    cap_attrs = {'Capacitance': 1, 'SizeCode': 2, 'RatedDCVoltageURdc': 3, 'PositiveTolerance': 4,
                 'NegativeTolerance': 5, 'TemperatureCharacteristicsCode': 6, 'MfrPartNumber': 7, 'input class': 8}
    res_attrs = {'Resistance': 11, 'SizeCode': 12, 'WorkingVoltage': 13, 'Tolerance': 14, 'RatedPowerDissipationP': 15,
                 'MfrPartNumber': 16, 'input class': 17}
    all_class_list = {'Capacitors': cap_attrs, 'Resistors': res_attrs}


    # 将不同的category映射到一个int上
    cap_attrs_dict = dict(zip(cap_attrs.values(), cap_attrs.keys()))
    res_attrs_dict = dict(zip(res_attrs.values(), res_attrs.keys()))
    all_attrs_dict = {**cap_attrs_dict, **res_attrs_dict}

    write_result = True
    set_seed(1024)
    # model_checkpoint = "albert-base-v2"
    # 在tokenizer里面用到了
    model_checkpoint = "prajjwal1/bert-tiny"
    # model_checkpoint = r'C:\Users\coldkiller\Desktop\supplyframe\checkpoint-3500'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print('torch.cuda.is_available()', device)


    # 在自己的本机上用的是 "prajjwal1/bert-tiny" 这个参数
    # 在服务器上用的是 "albert-base-v2" 这个参数
    # 记得调mask的参数

    config = {'num_labels1': 9, 'num_labels2': 8, 'model': "prajjwal1/bert-tiny"}
    # config = {'num_labels1': 9, 'num_labels2': 8, 'model': "albert-base-v2"}


    # Creating the loss function and optimizer
    LEARNING_RATE = 1e-5
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    NUM_WORKERS = 6
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # loss_function 是一个全局变量

    # 正式的
    main(num_epoch=100, save_epoch=1, logging_step=100, file_num=100, train_size=10000, test_size=10000)

