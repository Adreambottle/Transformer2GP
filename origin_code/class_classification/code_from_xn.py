#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sampling
import os
import pandas as pd
import logging
import sklearn
import time
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, trainer_callback
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import gc
import sys
import re
import warnings

warnings.filterwarnings('always')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# In[2]:


config_class_features = sampling.read_data("config/config-class-features.json")
config_class_name = sampling.read_data("config/config-class-name.json")
config_classinfo = sampling.read_data("config/config-classinfo.json")
config_numeric_fields = sampling.read_data("config/config-numeric-fields.json")
config_dynamic_units = sampling.read_data("config/config-dynamic-units.json")
pair_params = sampling.read_data("config/pair-params")
class_dict = {'Resistors': 1, 'Capacitors': 2, 'others': 0}


# In[3]:


# 19个resistors的文件*18,12个capacitors的文件*29，338个others文件总共374个文件,
def convert2str(label_list, item):
    tmp_string = ''

    shuffled_item = list(item.items())
    random.shuffle(shuffled_item)

    text, text_sep = "", ""
    deliminter_list = ['#', ',', '/', ';', ':', '-', '_', ' ']
    deliminter = random.sample(deliminter_list, 1)[0]

    for (key, val) in shuffled_item:
        if key == 'category' or key == 'labels' or key == 'description':
            continue
        elif key == 'class':
            label_list.append(val)
        else:
            if random.uniform(0, 1) > 0.1:  # 90% chance to use another deliminter
                tmp_string += str(val) + random.sample(deliminter_list, 1)[0]
            else:
                tmp_string += str(val) + deliminter

    output = re.sub("(" + "|".join(deliminter_list) + ")$", "", tmp_string)
    return output


def get_input_from_file(path, inputClass, data_num, text_list, label_list, file_num, train_flag):
    # path = "/Users/meron/Desktop/Supplyframe/SupplyFrame/class_classification/MultiData/cap/100_synthetic_cap_0.json"
    with open(path, 'r', encoding='utf-8') as f:
        standard_data = json.loads(f.read())
    print('sys.getsizeof(standard_data)', sys.getsizeof(standard_data))
    # sample_data = random.sample(standard_data,343*data_num)
    standard_data_length = len(standard_data)
    print('len of standard_data:', standard_data_length)

    if train_flag:
        lower_bound = file_num * data_num * 338
        upper_bound = min(standard_data_length, (file_num + 1) * data_num * 338)
        sample_data = standard_data[lower_bound:upper_bound]
    else:
        upper_bound = min(standard_data_length, data_num * 338)
        sample_data = standard_data[0:upper_bound]

    del standard_data
    gc.collect()
    for item in sample_data:
        text_list.append(item['description'])
        label_list.append(inputClass)
    return text_list, label_list


def get_input_from_sampling(input_size, train_flag, file_num):
    """
    input_size: the number of items to sample
    returns the input for simpletransformer
    """
    # 读取所有catogory文件
    path = os.getcwd()
    files = os.listdir('./formatData')  # 得到文件夹下的所有文件名称
    rs_list = []

    # input_size为1表示从每一个category抽取一个样本
    text_list = []
    label_list = []
    problematic = set()

    if train_flag:
        text_list, label_list = get_input_from_file('preprocess/standard_cap.json', 'Capacitors', input_size, text_list,
                                                    label_list, file_num, train_flag)
        text_list, label_list = get_input_from_file('preprocess/standard_res.json', 'Resistors', input_size, text_list,
                                                    label_list, file_num, train_flag)
    else:
        text_list, label_list = get_input_from_file('preprocess/standard_cap_test.json', 'Capacitors', input_size,
                                                    text_list, label_list, file_num, train_flag)
        text_list, label_list = get_input_from_file('preprocess/standard_res_test.json', 'Resistors', input_size,
                                                    text_list, label_list, file_num, train_flag)



    print('进入数据生成循环')
    for file in files:  # 遍历文件夹
        file_path = os.path.join(path, 'formatData/' + file)
        if os.path.isfile(file_path):  # 判断是否是文件夹，不是文件夹才打开
            rs = sampling.sampling(file, 0.6)
            if 'Resistors' in file:
                continue

            elif 'Capacitors' in file:
                continue

            else:
                for i in range(input_size):
                    try:
                        item = [item for item in rs.random_sampling()][0]
                        text_list.append(convert2str(label_list, item))
                    except Exception as e:
                        problematic.add(file)
                        continue

    # to shuffle input
    res = []
    for i in range(len(label_list)):
        res.append((text_list[i], label_list[i]))
    random.shuffle(res)
    for i in range(len(label_list)):
        text_list[i] = res[i][0]
        label_list[i] = res[i][1]

    print('problematic', problematic)
    print('query数量：', len(text_list))
    print('label数量：', len(label_list))
    print('sys.getsizeof(text_list)', sys.getsizeof(text_list))
    print('sys.getsizeof(label_list)', sys.getsizeof(label_list))

    #     with open('running_output.txt','a') as f:
    #         f.write('query数量:'+str(len(text_list))+'\n')
    #         f.write('label数量:'+str(len(label_list))+'\n')
    #         f.write('problematic:'+str(problematic)+'\n')
    return text_list, label_list


# a,b=get_input_from_sampling(1,True)
# for i in range(len(b)):
#     print((a[i],b[i]),'\n')


# In[4]:


# #保存列表，每行一个元素
# with open('input_example.txt','w',encoding='utf-8') as f:
#     c=[]
#     for i in range(len(b)):
#         c.append(b[i]+'       '+a[i])
#     f.write('\n'.join(c))


# In[5]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# In[6]:


def train_input_process(config_classinfo, model_checkpoint, train_size, class_dict, tokenizer, file_num):
    """
    """
    files = os.listdir('./class_inputData')

    #     with open('running_output.txt','w') as f:
    #         f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')

    # 准备训练数据
    print(f'train dataset{file_num}\n')
    if f'train_encodings{file_num}.pt' in files and f'train_labels{file_num}.csv' in files:
        print('read train dataset\n')
        train_encodings = torch.load(f'./class_inputData/train_encodings{file_num}.pt')
        train_labels = pd.read_csv(f'./class_inputData/train_labels{file_num}.csv', index_col=0)
        train_labels = train_labels['0'].to_list()
    else:
        print('generate train dataset\n')
        train_dataset, train_labels = get_input_from_sampling(train_size, train_flag=True, file_num=file_num)
        train_encodings = tokenizer(train_dataset, padding=True, truncation=False)

        pd.DataFrame(train_labels).to_csv(f'./class_inputData/train_labels{file_num}.csv')
        torch.save(train_encodings, f'./class_inputData/train_encodings{file_num}.pt')
        print('train dataset saved')

    # encode the labels
    train_labels_encoded = list(map(lambda x: ClassEncoder(x), train_labels))

    print('training dataset is ok\n')
    return train_encodings, train_labels_encoded


def test_input_process(config_classinfo, model_checkpoint, test_size, class_dict, tokenizer):
    """
    """
    #    print('test dataset')
    #    files= os.listdir('./class_inputData')

    #     if 'test_encodings.pt' in files and 'test_labels.csv' in files:
    #         print('read test dataset\n')
    #         test_encodings = torch.load('./class_inputData/test_encodings.pt')
    #         test_labels = pd.read_csv('./class_inputData/test_labels.csv',index_col=0)
    #         test_labels = test_labels['0'].to_list()
    #     else:
    print('generate test dataset\n')
    test_dataset, test_labels = get_input_from_sampling(test_size, train_flag=False, file_num=1)
    test_encodings = tokenizer(test_dataset, padding=True, truncation=False)

    #         torch.save(test_encodings, './class_inputData/test_encodings.pt')
    #         pd.DataFrame(test_labels).to_csv('./class_inputData/test_labels.csv')
    #         print('test dataset saved\n')

    # encode the labels
    test_labels_encoded = list(map(lambda x: ClassEncoder(x), test_labels))
    print('testing dataset is ok\n')
    return test_encodings, test_labels_encoded, test_dataset[:1000]


# In[7]:


class processDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[8]:


def ClassEncoder(Class):
    if Class == 'Resistors':
        return 1
    elif Class == 'Capacitors':
        return 2
    else:
        return 0


def ClassDecoder(Class):
    if Class == 1:
        return 'Resistors'
    elif Class == 2:
        return 'Capacitors'
    else:
        return 'Others'


# In[9]:


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=1)
    acc = accuracy_score(labels, preds)

    print('compute_metrics:', len(labels))
    class_result = {'accuracy': acc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall}

    #     with open('class_classification_report.json','a',errors='ignore') as f:
    #         json.dump(class_report,f,ensure_ascii=False, indent = 4)

    global write_result
    if write_result:
        class_preds = [ClassDecoder(item) for item in preds]
        class_labels = [ClassDecoder(item) for item in labels]
        # class_report = classification_report(class_labels, class_preds,output_dict=True)
        class_report = classification_report(class_labels, class_preds)
        with open('class_classification_report.txt', 'a') as f:
            f.write(class_report)

    #     with open('class_running_output.txt','w') as f:
    #         f.write('class_result:'+str(class_result)+'\n')

    return class_result


# In[10]:


def train(num_epoch=10, logging_steps=10, file_num=10, train_size=150, test_size=30):
    set_seed(1024)
    global write_result
    write_result = True
    model_checkpoint = "albert-base-v2"
    # model_checkpoint = "prajjwal1/bert-tiny"
    # model_checkpoint = r'C:\Users\coldkiller\Desktop\supplyframe\checkpoint-3500'

    gpu_available = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print('torch.cuda.is_available()', gpu_available)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    training_args = TrainingArguments(
        output_dir='./models',
        dataloader_num_workers=7,
        do_train=True,
        #        do_eval = True,
        #        evaluation_strategy = 'steps',
        learning_rate=1e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        num_train_epochs=1,
        logging_steps=logging_steps,
        #        save_steps=10000,
        no_cuda=not gpu_available,
        seed=1024,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=20,
        logging_dir='./logs',
        load_best_model_at_end=True,
        #        metric_for_best_model = 'eval_loss',
        #        greater_is_better = False,
        #        save_total_limit=5,
        disable_tqdm=True)

    # read real data
    with open(r'preprocess/description_with_label.json', 'r', errors='ignore', encoding='utf-8') as f:
        js = f.read()
        real_data = json.loads(js, strict=False)

    real_input = []
    real_labels = []
    for i in range(len(real_data)):
        if 'class' in real_data[i]:
            real_input.append(real_data[i]['description'])
            real_labels.append(ClassEncoder(real_data[i]['class']))

    print('real_input', len(real_input))
    real_encodings = tokenizer(real_input, padding=True, truncation=False)
    real_dataset = processDataset(real_encodings, real_labels)

    # training loop
    total_time = 0
    for j in range(num_epoch):
        print(f'\n\n%%%%%%%%%%%%%%epoch{j}%%%%%%%%%%%%%%\n\n')
        for i in range(file_num):
            print(f'\n\nbegin generate train_dataset{i}\n\n')
            train_data, train_labels_encoded = train_input_process(config_classinfo, model_checkpoint, train_size,
                                                                   class_dict, tokenizer, i)
            train_dataset = processDataset(train_data, train_labels_encoded)

            #         if i>0:
            #             trainer.args.learning_rate = tmp_learning_rate

            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=real_dataset
            )
            # trainer.add_callback(trainer_callback.EarlyStoppingCallback(early_stopping_patience=500,early_stopping_threshold=0.0001))

            trainer.args.output_dir = f'./models/model{j}'
            # Train the model
            print('Training begins\n')
            start = time.time()
            trainer.train()
            end = time.time()
            print(f"training time: {end - start}")
            total_time += (end - start)

            # 为下一批训练数据做初始化
            trainer.args.warmup_steps = 0
            # tmp_learning_rate = trainer.args.learning_rate
            gc.collect()

        # 保存模型
        trainer.save_model()

        with open('class_classification_report.txt', 'a') as f:
            f.write(f'%%%%%%%%%%epoch {j}%%%%%%%%%%%\n')
        real_result = trainer.predict(real_dataset)
        print('real_result:  ', real_result.metrics)
        with open('validation_report.txt', 'a') as f:
            f.write(f'epoch {j}:{real_result.metrics}\n')
    #     with open('albert_structure.txt','w') as f:
    #         f.write(str(trainer.model))

    print('\n\ntotal_training_time:', total_time)
    # generate test data
    test_data, test_labels_encoded, sample_data = test_input_process(config_classinfo, model_checkpoint, test_size,
                                                                     class_dict, tokenizer)
    test_dataset = processDataset(test_data, test_labels_encoded)

    # show result
    #     global write_result
    #     write_result = True
    with open('class_classification_report.txt', 'a') as f:
        f.write('~~~~~~~~~~training result:~~~~~~~~~~\n')
    train_result = trainer.predict(train_dataset).metrics
    print('train_result:  ', train_result)

    with open('class_classification_report.txt', 'a') as f:
        f.write('~~~~~~~~~~testing result:~~~~~~~~~~\n')
    test_result = trainer.predict(test_dataset)
    print('test_result:  ', test_result.metrics)

    with open('class_classification_report.txt', 'a') as f:
        f.write('~~~~~~~~~~validation result:~~~~~~~~~~\n')
    real_result = trainer.predict(real_dataset)
    print('real_result:  ', real_result.metrics)

    # save result
    test_preds = [ClassDecoder(item) for item in test_result.predictions.argmax(-1)[:1000]]
    test_labels = [ClassDecoder(item) for item in test_labels_encoded[:1000]]
    test_res = pd.DataFrame({"description": sample_data, "true_labels": test_labels, "predicted_labels": test_preds})
    test_res.to_csv('test_description_result.csv', encoding='utf_8_sig')

    real_preds = [ClassDecoder(item) for item in real_result.predictions.argmax(-1)]
    real_labels = [ClassDecoder(item) for item in real_labels]
    real_res = pd.DataFrame({"description": real_input, "true_labels": real_labels, "predicted_labels": real_preds})
    real_res.to_csv('real_description_result.csv', encoding='utf_8_sig')


#     with open('running_output.txt','a') as f:
#         f.write(f"training time: {end - start}"+'\n')
#         f.write('train_dataset'+str(train_result)+'\n')
#         f.write('test_dataset'+str(test_result)+'\n')

#     with open('real_description_output.json','w',errors='ignore') as f:
#         json.dump(class_result,f,ensure_ascii=False, indent = 4)


# In[11]:


if __name__ == "__main__":
    write_result = True
    # train(num_epoch=50, logging_steps=50, file_num=30, train_size=100, test_size=1000)
    train(num_epoch=1, logging_steps=50, file_num=30, train_size=10, test_size=10)


# In[12]:


# 改比例，训练数据数量，epoch,dataloader_num_workers,模型


# In[13]:


# albert50万
# training_args = TrainingArguments(
#     output_dir='./results',
#     dataloader_num_workers=7,
#     do_train = True,
#     learning_rate=1e-5,
#     weight_decay=0.01,
#     adam_beta1=0.9,
#     adam_beta2=0.999,
#     adam_epsilon=1e-8,
#     num_train_epochs=10,
#     logging_steps=10,
#     save_steps=10000,
#     no_cuda= not gpu_available,
#     seed=1024,
#     per_device_train_batch_size=64,
#     per_device_eval_batch_size=64,
#     warmup_steps=20,
#     logging_dir='./logs',
#     load_best_model_at_end=True,
#     save_total_limit=5,
#     disable_tqdm=True)


# In[14]:


# ALBERT150万
# training_args = TrainingArguments(
#     output_dir='./results',
#     dataloader_num_workers=7,
#     do_train = True,
#     do_eval = True,
#     evaluation_strategy = 'steps',
#     learning_rate=1e-5,
#     weight_decay=0.01,
#     adam_beta1=0.9,
#     adam_beta2=0.999,
#     adam_epsilon=1e-8,
#     num_train_epochs=num_train_epochs,
#     logging_steps=logging_steps,
#     save_steps=10000,
#     no_cuda= not gpu_available,
#     seed=1024,
#     per_device_train_batch_size=64,
#     per_device_eval_batch_size=64,
#     warmup_steps=20,
#     logging_dir='./logs',
#     load_best_model_at_end=True,
#     metric_for_best_model = 'eval_loss',
#     greater_is_better = False,
#     save_total_limit=5,
#     disable_tqdm=True)


# In[15]:


# bert-tiny
# training_args = TrainingArguments(
#         output_dir='./results',
#         dataloader_num_workers=7,
#         do_train = True,
#         learning_rate=1e-5,
#         weight_decay=0.01,
#         adam_beta1=0.9,
#         adam_beta2=0.999,
#         adam_epsilon=1e-8,
#         num_train_epochs=100,
#         logging_steps=100,
#         save_steps=500000,
#         no_cuda= not gpu_available,
#         seed=1024,
#         per_device_train_batch_size=256,
#         per_device_eval_batch_size=256,
#         warmup_steps=5,
#         logging_dir='./logs',
#         load_best_model_at_end=True,
#         save_total_limit=5,
#         disable_tqdm=True
#     )

