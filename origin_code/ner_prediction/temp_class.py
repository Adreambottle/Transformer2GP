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


def insert_deliminter(item):
    """
    inserts deliminters to description
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


def get_input_from_file_single(path, inputClass, data_num, text_list, label_list, file_num, train_flag):
    """
    这个函数是用来从 Cap 和 Res 里面读取原始数据，生成一个分割好的 data

    path:        现在是: "preprocess/standard_cap.json"
                        "preprocess/standard_res.json"
    inputClass:  是 "Capacitors", "Resistors", 还是"Others"
    data_num:    是数据的个数，每个文件里面有多少条数据
    file_num:    是生成几个拆分好的文件包
    train_flag:  是用于train还是test
    text_list:   把生成的sample的'description'添加到text_list
    label_list:  把生成的sample的 class label 添加到label_list
    """

    with open(path, 'r', encoding='utf-8') as f:
        standard_data = json.loads(f.read())
        # standard_data 就是一个list，是读取'preprocess/standard_cap.json'
        # 这里需要修改，暂时长度是969
    print('\tsys.getsizeof(standard_data)', sys.getsizeof(standard_data))
    # 这个是standard_data的数据大小

    standard_data_length = len(standard_data)
    print('\tlen of standard_data:', standard_data_length)
    # 这个是standard_data的数据长度

    if train_flag:
        # 如果是train的话，生成一个sample_data
        lower_bound = file_num * data_num * 338  # 数据集的下限
        upper_bound = min(standard_data_length, (file_num + 1) * data_num * 338)  # 数据集的上限
        sample_data = standard_data[lower_bound:upper_bound]  # 选取standard_data中的这个部分

    else:
        # 如果是test的话，
        upper_bound = min(standard_data_length, data_num * 338)
        sample_data = standard_data[0:upper_bound]  # 测试集只需要上限就好

    del standard_data  # 生成之后就直接删掉原始的数据集
    gc.collect()

    for item in sample_data:
        text_list.append(item['description'])  # 向text_list中添加'description'
        label_list.append(inputClass)  # 向label_list中添加inputClass
    return text_list, label_list


def get_input_from_file(path, inputClass, data_num, text_list, label_list, file_num, train_flag):
    """
    这个函数是用来从 Cap 和 Res 里面读取原始数据，生成一个分割好的 data

    path:        现在是: "preprocess/standard_cap.json"
                        "preprocess/standard_res.json"
    inputClass:  是 "Capacitors", "Resistors", 还是"Others"
    data_num:    是数据的个数，每个文件里面有多少条数据
    file_num:    是生成几个拆分好的文件包
    train_flag:  是用于train还是test
    text_list:   把生成的sample的'description'添加到text_list
    label_list:  把生成的sample的 class label 添加到label_list
    """

    # path = "/Users/meron/Desktop/Supplyframe/SupplyFrame/class_classification/MultiData/cap"
    files = os.listdir(path)
    files_json = []
    for file in files:
        if file[-5:] == ".json":
            files_json.append(file)

    files_json_num = 0
    for file in files_json:

        # file = files_json[0]
        print(f"\t读取第{files_json_num}个原始数据")
        files_json_num += 1
        with open(path + '/' + file, 'r', encoding='utf-8') as f:
            standard_data = json.loads(f.read())
            # standard_data 就是一个list，是读取'preprocess/standard_cap.json'
            # 这里需要修改，暂时长度是969
        print('\tsys.getsizeof(standard_data)', sys.getsizeof(standard_data))
        # 这个是standard_data的数据大小

        standard_data_length = len(standard_data)
        print('\tlen of standard_data:', standard_data_length)
        # 这个是standard_data的数据长度

        if train_flag:
            # 如果是train的话，生成一个sample_data
            lower_bound = file_num * data_num * 338  # 数据集的下限
            upper_bound = min(standard_data_length, (file_num + 1) * data_num * 338)  # 数据集的上限
            sample_data = standard_data[lower_bound:upper_bound]  # 选取standard_data中的这个部分

        else:
            # 如果是test的话
            upper_bound = min(standard_data_length, data_num * 338)
            sample_data = standard_data[0:upper_bound]  # 测试集只需要上限就好

        del standard_data  # 生成之后就直接删掉原始的数据集
        gc.collect()
        for item in sample_data:
            # item = sample_data[0]
            item = insert_deliminter(item)
            text_list.append(item['description'])  # 向text_list中添加'description'
            label_list.append(inputClass)  # 向label_list中添加inputClass
    return text_list, label_list


def get_input_from_sampling(input_size, train_flag, file_num):
    """
    input_size: the number of items to sample
    returns the input for simpletransformer
    感觉这部分就是读取 Cat 和 Res 的数据，然后自动生成 Others 的数据
    """
    # 读取所有catogory文件
    path = os.getcwd()  # path的地址是 '.SupplyFrame/class_classification'
    files_formatData = os.listdir('./formatData')  # 获取formatData这个文件夹下的所有文件

    # 得到文件夹下的所有文件名称
    rs_list = []

    # input_size为1表示从每一个category抽取一个样本
    text_list = []
    label_list = []
    problematic = set()

    cap_path = '/data/bob/synthetic/cap/'
    res_path = '/data/bob/synthetic/res/'

    # 这里是用来读取 Cap 和 Res 的数据的
    # 这里的 text_list 是一个列表，里层是一个嵌套列表，需要把里层的列表合成一个string
    # 这里的 label_list 是一个列表，其中只是 'Capacitors' 在前，'Resistors' 在后，
    if train_flag:

        # 读取的是preprocess中的数据，如果是train_flag == 1的话，读取的是train的数据
        text_list, label_list = get_input_from_file(cap_path, 'Capacitors', input_size, text_list,
                                                    label_list, file_num, train_flag)
        text_list, label_list = get_input_from_file(res_path, 'Resistors', input_size, text_list,
                                                    label_list, file_num, train_flag)
    else:
        # 如果是train_flag != 1的话，读取的是test的数据
        text_list, label_list = get_input_from_file(cap_path, 'Capacitors', input_size,
                                                    text_list, label_list, file_num, train_flag)
        text_list, label_list = get_input_from_file(res_path, 'Resistors', input_size,
                                                    text_list, label_list, file_num, train_flag)

    print('进入数据生成循环')
    files_formatData_json = []
    for file in files_formatData:
        if file[-5:] == ".json":
            files_formatData_json.append(file)

    for file in files_formatData_json:  # 遍历文件夹
        # file = files_formatData[0]

        file_path = os.path.join(path, 'formatData/' + file)
        if os.path.isfile(file_path):  # 判断是否是文件夹，不是文件夹才打开

            # 这个 0.6 是用来做什么的
            rs = sampling.sampling(file, 0.6)

            # 如果是 'Resistors' 或 'Capacitors' 则略过
            if 'Resistors' in file:
                continue

            elif 'Capacitors' in file:
                continue

            else:
                for i in range(input_size):

                    try:
                        item = [item for item in rs.random_sampling()][0]
                        text_new = convert2str(label_list, item)  # 将 label 转换为 string
                        text_list.append(text_new)
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

    print('\tproblematic', problematic)
    print('\ttext数量：', len(text_list))
    print('\tlabel数量：', len(label_list))
    print('\tsys.getsizeof(text_list)', sys.getsizeof(text_list))
    print('\tsys.getsizeof(label_list)', sys.getsizeof(label_list))

    return text_list, label_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# In[6]:


def train_input_process(config_classinfo, model_checkpoint, train_size, class_dict, tokenizer, file_num):
    """
    这部分是用于生成train_dataset，需要生成很多个dataset
    返回的是train_data, train_labels_encoded
    """

    files = os.listdir('./class_inputData')
    # 查看工作路径下有什么文件，返回一个list

    # 准备训练数据
    print(f'处理 train dataset{file_num}\n')

    if f'train_encodings{file_num}.pt' in files and f'train_labels{file_num}.csv' in files:
        # 如果存在"train_encodings{file_num}.pt" 和 "train_labels{file_num}.csv" 这两个数据的话，直接读取
        print(f'\t有数据存在，读取数据train_encodings{file_num}.pt\n')
        train_encodings = torch.load(f'./class_inputData/train_encodings{file_num}.pt')
        train_labels = pd.read_csv(f'./class_inputData/train_labels{file_num}.csv', index_col=0)
        train_labels = train_labels['0'].to_list()
    else:
        print(f'\t无数据存在，生成数据train_encodings{file_num}.pt\n')
        train_dataset, train_labels = get_input_from_sampling(train_size, train_flag=True,
                                                              file_num=file_num)           # get_input_from_sampling 是一个自己编写的数据生成函数
        train_encodings = tokenizer(train_dataset, padding=True, truncation=False)

        pd.DataFrame(train_labels).to_csv(f'./class_inputData/train_labels{file_num}.csv')  # 将train_labels改写成csv储存起来
        torch.save(train_encodings, f'./class_inputData/train_encodings{file_num}.pt')      # 将train_encodings用pt的格式存储起来
        print('\ttrain_dataset{file_num} saved')

    # encode the labels
    train_labels_encoded = list(map(lambda x: ClassEncoder(x), train_labels))

    print('training dataset is ok\n')
    return train_encodings, train_labels_encoded


def test_input_process(config_classinfo, model_checkpoint, test_size, class_dict, tokenizer):
    """
    还不知道这个函数是干什么的

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
    """
    这个类是继承自torch.utils.data.Dataset
    用于将text生成的encoding部分和label部分输入给Transformer模型
    """

    def __init__(self, encodings, labels):
        """
        生成函数里面需要放入 encodings 和 labels
        encodings 是已经 tokenlization 之后的结果
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)  # 返回label的长度


# In[8]:


def ClassEncoder(Class):
    """
    将class标签进行分类
    """
    if Class == 'Resistors':
        return 1
    elif Class == 'Capacitors':
        return 2
    else:
        return 0


def ClassDecoder(Class):
    """
    将class反向翻译
    """
    if Class == 1:
        return 'Resistors'
    elif Class == 2:
        return 'Capacitors'
    else:
        return 'Others'


# In[9]:


def compute_metrics(pred):
    """
    这个数据好像是用来评价的
    """
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
    model_checkpoint = "albert-base-v2"  # 采用的模型
    # model_checkpoint = "prajjwal1/bert-tiny"         # 备用的模型，什么是checkpoint
    # model_checkpoint = r'C:\Users\coldkiller\Desktop\supplyframe\checkpoint-3500'

    gpu_available = torch.cuda.is_available()                    # 查看GPU加速是否在运行
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # 将文字 tokenlization，这个是掉包的函数

    print('torch.cuda.is_available()', gpu_available)            # 如果 GPU 是否可以用

    # 应该是预训练应用的模型
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
    # trainer 的参数
    training_args = TrainingArguments(
        output_dir='./models',                     # 模型储存的地址
        dataloader_num_workers=7,                  # 这个是分布式加载数据的，用了7个
        do_train=True,
        #        do_eval = True,
        #        evaluation_strategy = 'steps',
        learning_rate=1e-5,                        # 学习率
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


    # 第一步读取真实的数据
    # 开始读数据，这里的是真实的数据，暂时不知道这部分是用来做什么的
    with open(r'preprocess/description_with_label.json', 'r', errors='ignore', encoding='utf-8') as f:
        js = f.read()
        real_data = json.loads(js, strict=False)
        # real_data 是一个list，list中的每一个元素是一个dictionary


    # 将不同类型的数据混淆
    real_input = []  # 生成真实数据的input
    real_labels = []  # 生成真实数据的label

    # 从数据里面筛选出有 "class" 和 "description" 的
    for i in range(len(real_data)):
        if 'class' in real_data[i]:
            real_input.append(real_data[i]['description'])
            real_labels.append(ClassEncoder(real_data[i]['class']))

    print('真实数据的长度', len(real_input))
    real_encodings = tokenizer(real_input, padding=True, truncation=False)  # tokenizer 是模型参数
    # real_encodings 是一个list
    # Encoding(num_tokens=44, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])

    real_dataset = processDataset(real_encodings, real_labels)
    # processDataset 是一个class，将encodings和labels合在一起






    # 第二步：生成train_data
    total_time = 0
    print("开始进入train数据阶段")
    # 先对每个epoch进行循环
    for j in range(num_epoch):

        # j 代表的是第几个epoch
        print(f'\n\n%%%%%%%%%%%%%%epoch{j}%%%%%%%%%%%%%%\n\n')

        # 再对每个batch进行循环，每个batch的数据都是尽可能占满整个显存
        # 每个batch都要运行一遍trainer
        for i in range(file_num):
            print(f'\n\n开始生成 train_dataset{i}\n\n')

            # 处理train的数据
            # processDataset()，将encodings和labels合在一起，传入 trainer 中
            train_data, train_labels_encoded = train_input_process(config_classinfo,
                                                                   model_checkpoint,
                                                                   train_size,
                                                                   class_dict,
                                                                   tokenizer,
                                                                   i)

            train_dataset = processDataset(train_data, train_labels_encoded)


            # trainer是直接调用transformer包中的训练器
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
            print(f"training time for one batch: {end - start}")
            total_time += (end - start)

            # 为下一批训练数据做初始化
            trainer.args.warmup_steps = 0
            # tmp_learning_rate = trainer.args.learning_rate
            gc.collect()

        # 保存模型，这是调用的trainer的一个参数
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

    train(num_epoch=30, logging_steps=50, file_num=30, train_size=100, test_size=1000)
    # train(num_epoch=1, logging_steps=5, file_num=1, train_size=1, test_size=1)
