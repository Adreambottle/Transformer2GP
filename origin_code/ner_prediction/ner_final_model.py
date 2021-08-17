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
from transformers import BertModel,AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,trainer_callback,DataCollatorForTokenClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,classification_report
import json
import gc
import sys
import re
from seqeval.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[2]:


def get_input_from_file(path,data_num,text_list,label_list,class_list,file_num,train_flag):
    with open(path,'r', encoding='utf-8') as f:
        standard_data = json.loads(f.read())
    #print('sys.getsizeof(standard_data)',sys.getsizeof(standard_data))
    #sample_data = random.sample(standard_data,343*data_num)
    standard_data_length = len(standard_data)
    print('len of standard_data:',standard_data_length)
    
    if train_flag:
        lower_bound = file_num*data_num*1
        upper_bound = min(standard_data_length,(file_num+1)*data_num*1)
        sample_data = standard_data[lower_bound:upper_bound]
    else:
        upper_bound = min(standard_data_length,data_num*1)
        sample_data = standard_data[0:upper_bound]
        
    del standard_data
    gc.collect()
    
    for item in sample_data:
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
    return text_list,label_list,class_list
        
def get_input_from_sampling(input_size,train_flag,file_num):
    """
    input_size: the number of items to sample
    returns the input for simpletransformer
    """
    # 将抽取到的结果转化为输入格式
    text_list = []
    label_list = []
    class_list = []
    
    if train_flag:
        text_list,label_list,class_list = get_input_from_file('preprocess/standard_cap.json',input_size,text_list,label_list,class_list,file_num,train_flag)
        text_list,label_list,class_list = get_input_from_file('preprocess/standard_res.json',input_size,text_list,label_list,class_list,file_num,train_flag)
    else:
        text_list,label_list,class_list = get_input_from_file('preprocess/standard_cap_test.json',input_size,text_list,label_list,class_list,file_num,train_flag)
        text_list,label_list,class_list = get_input_from_file('preprocess/standard_res_test.json',input_size,text_list,label_list,class_list,file_num,train_flag)

    #to shuffle input
    res=[]
    for i in range(len(label_list)):
        res.append((text_list[i],label_list[i],class_list[i]))
    random.shuffle(res)
    for i in range(len(label_list)):
        text_list[i]=res[i][0]
        label_list[i]=res[i][1]
        class_list[i]=res[i][2]
        
        
    print('query数量：',len(text_list))
    print('label数量：',len(label_list))
    print('class数量：',len(class_list))
    
#     print('sys.getsizeof(text_list)',sys.getsizeof(text_list))
#     print('sys.getsizeof(label_list)',sys.getsizeof(label_list))
    
    return text_list,label_list,class_list

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


def tokenize_and_align_labels(text,labels_before_split,tokenizer):
    tokenized_inputs = tokenizer(text, is_split_into_words=True,add_special_tokens=False,
                        padding=True,truncation=False,return_tensors="pt")
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
    return tokenized_inputs,labels

def show_tokenizer_result(text,labels_before_split,tokenizer):
    tokenized_inputs,labels = tokenize_and_align_labels(text,labels_before_split,tokenizer)
    for i in range(len(labels_before_split)):
        print('text:',text[i])
        print('tokenized result:',tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][i]))
        #print('tokenized_inputs:',tokenized_inputs["input_ids"][i])
        
        print('labels_before_split:',labels_before_split[i])
        print('labels:',labels[i])
        print('\n%%%%%%%%%%%%%%%%%%%%%\n')

#show_tokenizer_result(a[:10],b[:10],tokenizer)


# In[5]:


# 读文件里面的数据转化为二维列表
def read_list(filename):
    file1 = open(filename+".txt", "r")
    list_row =file1.readlines()
    list_source = []
    for i in range(len(list_row)):
        column_list = list_row[i].strip().split("\t")  # 每一行split后是一个列表
        list_source.append(column_list)                # 在末尾追加到list_source
    file1.close()
    return list_source

#保存二维列表到文件
def save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()


# In[6]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def attrEncoder(all_class_list,item_class,attr):
    if item_class in all_class_list and attr in all_class_list[item_class]:
        if item_class == 'Capacitors':
            return all_class_list[item_class][attr]
        if item_class == 'Resistors':
            return all_class_list[item_class][attr]-10
    if attr == '-100':
        return -100
    return 0

def attrDecoder(item_class,attr):
    if attr==0 or attr==-100:
        return 'others'
    else:
        if item_class == 'Resistors':
            attr+=10
        global all_attrs_dict
        return all_attrs_dict[attr]

#attrDecoder('Resistors',3)


# In[7]:


# train_encodings = tokenizer(train_dataset, is_split_into_words=True,add_special_tokens=False,padding=True,truncation=True)
# for i in range(len(train_encodings)):
#     print('tokenized result:',tokenizer.convert_ids_to_tokens(train_encodings["input_ids"][i]))


# In[8]:


def train_input_process(train_size,tokenizer,file_num):
    """
    """
    files= os.listdir('./inputData')

#     with open('running_output.txt','w') as f:
#         f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
    
    #准备训练数据
    print(f'train dataset{file_num}\n')
    if f'train_encodings{file_num}.pt' in files and f'train_labels{file_num}.txt' in files and f'train_class{file_num}.csv' in files:
        print('read train dataset\n')
        train_encodings = torch.load(f'./inputData/train_encodings{file_num}.pt')
        train_labels = read_list(f'./inputData/train_labels{file_num}')
        train_class = pd.read_csv(f'./inputData/train_class{file_num}.csv',index_col=0)
        train_class = train_class['0'].to_list()
    else:
        print('generate train dataset\n')
        train_dataset,train_labels,train_class = get_input_from_sampling(train_size,train_flag=True,file_num=file_num)
        #train_encodings = tokenizer(train_dataset, is_split_into_words=True,add_special_tokens=False,
#                        padding=True,truncation=False,return_tensors="pt")
        train_encodings,train_labels = tokenize_and_align_labels(train_dataset,train_labels,tokenizer)

        pd.DataFrame(train_class).to_csv(f'./inputData/train_class{file_num}.csv')
        save_list(train_labels,f'./inputData/train_labels{file_num}')
        torch.save(train_encodings, f'./inputData/train_encodings{file_num}.pt')
        print('train dataset saved')
        
    # encode the labels
    global all_class_list
    train_labels_encoded = []
    for i in range(len(train_labels)):
#        train_labels_encoded[i] = list(map(lambda x,y,z:attrEncoder(x,y,z),all_class_list,train_class[i],train_labels[i]))
        train_labels_encoded.append([attrEncoder(all_class_list,train_class[i],train_label) for train_label in train_labels[i]])
    
    print('training dataset is ok\n')
    return train_encodings,train_labels_encoded,train_class

def test_input_process(test_size,tokenizer):
    """
    """
    print('generate test dataset\n')
    test_dataset,test_labels,test_class = get_input_from_sampling(test_size,train_flag=False,file_num=1)
    test_encodings,test_labels = tokenize_and_align_labels(test_dataset,test_labels,tokenizer)

    # encode the labels
    global all_class_list
    test_labels_encoded = []
    for i in range(len(test_labels)):
        test_labels_encoded.append([attrEncoder(all_class_list,test_class[i],test_label) for test_label in test_labels[i]]) 
                  
    print('testing dataset is ok\n')
    return test_encodings,test_labels_encoded,test_class,test_dataset[:1000],test_labels

#train_data,train_labels_encoded,train_class = train_input_process(model_checkpoint,3,tokenizer,1)
#te_data,te_labels_encoded,te_class,test_sample,test_labels = test_input_process(model_checkpoint,10,tokenizer)


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
    def __init__(self, encodings, labels,classes):
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
    def __init__(self,config):
        super(nerClass, self).__init__()
        self.num_labels1 = config['num_labels1']
        self.num_labels2 = config['num_labels2']
        self.l1 = BertModel.from_pretrained(config['model'])
        
#         self.pre_classifier = torch.nn.Linear(128, 128)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier1 = torch.nn.Linear(128, config['num_labels1'])
        self.classifier2 = torch.nn.Linear(128, config['num_labels2'])
#        self.init_weights()

    def forward(self, input_ids, attention_mask=None,targets=None,classes=None,device=None):
#         output_1 = self.l1(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
#                             attention_mask=attention_mask, head_mask=head_mask)
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
#        print('hidden_state.size()',hidden_state[:,0].size())
#        pooler = self.pre_classifier(pooler)
#        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(hidden_state)
#        print('pooler.size()',pooler.size())
        
        mask1 = torch.eq(classes,1)
        mask2 = torch.eq(classes,2)
#        print('pooler[mask1].size()',pooler[mask1].size())
#        print('pooler[mask2].size()',pooler[mask2].size())
        output1 = self.classifier1(pooler[mask1])
        output2 = self.classifier2(pooler[mask2])
#         print('output1.size()',output1.size())
#         print('output2.size()',output2.size())
        
#        output1 = output1[mask1]
        target1 = targets[mask1]
#         print('target1.size()',target1.size())
        
#        output2 = output2[mask2]  
        target2 = targets[mask2]
#         print('target2.size()',target2.size())
        
        if output1.size()[0]!=0:
            if attention_mask[mask1] is not None:
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
            loss1 = torch.tensor(0.0,requires_grad=True,device=device)
        
        if output2.size()[0]!=0:
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
            loss2 = torch.tensor(0.0,requires_grad=True,device=device)
#         print(loss1)
#        print(type(loss1))
        return output1,output2,mask1,mask2,loss1,loss2


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


def train(model,epoch,cur_file,file_num,logging_step=1,training_loader=None,optimizer=None):
    tr_loss1 = 0
    tr_loss2 = 0
    tr_total_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _,data in enumerate(training_loader, start=0):
        ids = data['input_ids'].to(device, dtype = torch.long)
        attention_mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['labels'].to(device, dtype = torch.long)
        classes = data['classes'].to(device, dtype = torch.long)

        #对梯度进行初始化
        optimizer.zero_grad()
        
        #前向传播
        output1,output2,mask1,mask2,loss1,loss2 = model(ids, attention_mask,targets,classes,device)

        #保存每个epoch最后一个file的结果
        if cur_file == file_num:
            if _ == 0 :
                out_tensor1 = output1.clone()
                out_tensor2 = output2.clone()
                target_tensor1 = targets[mask1].clone()
                target_tensor2 = targets[mask2].clone()
            else:
                out_tensor1 = torch.cat((out_tensor1,output1),0)
                out_tensor2 = torch.cat((out_tensor2,output2),0)
                target_tensor1 = torch.cat((target_tensor1,targets[mask1]),0)
                target_tensor2 = torch.cat((target_tensor2,targets[mask2]),0)
                

        #print(loss1.item(), loss2.item())
        tr_loss1 += loss1.item()
        tr_loss2 += loss2.item()
        tr_total_loss += loss1.item()+loss2.item()

        #反向传播
        loss1.backward(retain_graph=True)
        loss2.backward()

        #对网络的参数进行更新
        optimizer.step()
        
        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        #logging
        if _%logging_step==0:
            loss_step = tr_total_loss/nb_tr_steps
            loss_step1 = tr_loss1/nb_tr_steps
            loss_step2 = tr_loss2/nb_tr_steps
            #accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss step{_}: total loss:{loss_step},cap loss:{loss_step1},res loss:{loss_step2}")
            #print(f"Training Accuracy per 5000 steps: {accu_step}")

    # evaluate
    if cur_file == file_num:
        print('evaluate training result\n')
        with open('class_classification_report.txt','a') as f: 
            f.write(f'\n~~~~~~~~~~training epoch{epoch} result:~~~~~~~~~~\n')
        true_labels1, true_predictions1 = compute_metrics(True,target_tensor1.cpu().data.numpy().tolist(),out_tensor1.argmax(-1).cpu().data.numpy().tolist(),'Capacitors')
        true_labels2, true_predictions2 = compute_metrics(True,target_tensor2.cpu().data.numpy().tolist(),out_tensor2.argmax(-1).cpu().data.numpy().tolist(),'Resistors')

    #print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_total_loss/nb_tr_steps
#    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
#    print(f"Training Accuracy Epoch: {epoch_accu}")
    return model
#model = train(1,1)


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


def compute_metrics(train_flag,labels,preds,classes):
#    print(labels)
    # Remove ignored index (special tokens)
    global all_class_list
    true_predictions = [[attrDecoder(classes,p) for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(preds, labels)]
    true_labels = [[attrDecoder(classes,l) for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(preds, labels)]
    
#     print(true_predictions)
#     print(true_labels)
    results = compute(predictions=true_predictions, references=true_labels)
    #print(results)
    
    #对result进行处理，以写入文件
    for k,v in results.items():
        if isinstance(v,dict):
            v['number']=str(v['number'])

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
        #class_report = classification_report(class_labels, class_preds,output_dict=True)
        #class_report = classification_report(class_labels, class_preds)
#         with open('class_classification_report.txt','a') as f: 
#             f.write(str(results))

        with open('class_classification_report.json','a',errors='ignore') as f: 
            json.dump(results,f,ensure_ascii=False, indent = 4) 
            
        res = {"precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"]}
        if train_flag:
            print('train_result:',res)
            with open('class_classification_report.txt','a') as f: 
                f.write(str(res)+'\n')
        else:
            print('test_result:',res)
            with open('class_classification_report.txt','a') as f: 
                f.write(str(res)+'\n')

    return true_labels, true_predictions


# In[22]:


def test(model,logging_step,testing_loader,optimizer,test_class_part,sample_data):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    tr_loss1 = 0
    tr_loss2 = 0
    tr_total_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):

            ids = data['input_ids'].to(device, dtype = torch.long)
            attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)
            classes = data['classes'].to(device, dtype = torch.long)
            #print(classes.size())
            
            #前向传播
            output1,output2,mask1,mask2,loss1,loss2 = model(ids, attention_mask,targets,classes,device)
            
            if _ == 0 :
                out_tensor1 = output1.clone()
                out_tensor2 = output2.clone()
                target_tensor1 = targets[mask1].clone()
                target_tensor2 = targets[mask2].clone()
            else:
                out_tensor1 = torch.cat((out_tensor1,output1),0)
                out_tensor2 = torch.cat((out_tensor2,output2),0)
                target_tensor1 = torch.cat((target_tensor1,targets[mask1]),0)
                target_tensor2 = torch.cat((target_tensor2,targets[mask2]),0)
            
            
#             for item in ids[mask1]:
#                 res=tokenizer.convert_ids_to_tokens(item)
#                 #print(res)
#                 res=tokenizer.convert_tokens_to_string(res)
#                 res=res.replace('[PAD]','')
#                 input.append(res.strip())
                
            #print(loss1.item(), loss2.item())
            tr_loss1 += loss1.item()
            tr_loss2 += loss2.item()
            tr_total_loss += loss1.item()+loss2.item()

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            #logging
            if _%logging_step==0:
                loss_step = tr_total_loss/nb_tr_steps
                loss_step1 = tr_loss1/nb_tr_steps
                loss_step2 = tr_loss2/nb_tr_steps
                #accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Testing Loss per step: total loss:{loss_step},cap loss:{loss_step1},res loss:{loss_step2}")
                #print(f"Training Accuracy per 5000 steps: {accu_step}")
#     print(target_tensor1.size())
#     print(out_tensor1.argmax(-1).size())
    with open('class_classification_report.txt','a') as f: 
        f.write('\n~~~~~~~~~~testing result:~~~~~~~~~~\n')
    true_labels1, true_predictions1 = compute_metrics(False,target_tensor1.cpu().data.numpy().tolist(),out_tensor1.argmax(-1).cpu().data.numpy().tolist(),'Capacitors')
    true_labels2, true_predictions2 = compute_metrics(False,target_tensor2.cpu().data.numpy().tolist(),out_tensor2.argmax(-1).cpu().data.numpy().tolist(),'Resistors')
    
    cap_input=[v for i,v in zip(test_class_part,sample_data) if i=='Capacitors']
    res_input=[v for i,v in zip(test_class_part,sample_data) if i=='Resistors']
    
    min_len1 = min(len(true_labels1),len(cap_input))
    min_len2 = min(len(true_labels2),len(res_input))

    result_desc = cap_input[:min_len1]+res_input[:min_len2]
    result_labels = true_labels1[:min_len1]+true_labels2[:min_len2]
    result_predictions = true_predictions1[:min_len1]+true_predictions2[:min_len2]
    
    join_original_input = pd.DataFrame({"description":result_desc,"labels":result_labels,'predictions':result_predictions})
    join_original_input.to_csv('test_description_result.csv',encoding='utf_8_sig')
    
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


def main(num_epoch=10,save_epoch=2,logging_step=10,file_num=10,train_size=150,test_size=30):
    model = nerClass(config)
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': NUM_WORKERS}

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': NUM_WORKERS}
    
    # training loop
    total_time = 0
    for j in range(num_epoch):
        print(f'\n\n%%%%%%%%%%%%%%epoch{j}%%%%%%%%%%%%%%\n\n')
        for i in range(file_num):
            print(f'\n\nepoch{j} begin generate train_dataset{i}\n\n')
            train_data,train_labels_encoded,train_class = train_input_process(train_size=train_size,tokenizer=tokenizer,file_num=i)
            train_class_encoded = list(map(lambda x:ClassEncoder(x),train_class))
            train_dataset = processDataset(train_data, train_labels_encoded,train_class_encoded)
            training_loader = DataLoader(train_dataset, **train_params)

            # Train the model
            print('Training begins\n')
            start = time.time()
            
            model = train(model=model,epoch=j,cur_file=i+1,file_num=file_num,logging_step=logging_step,training_loader=training_loader,optimizer=optimizer)
            
            end = time.time()
            print(f"training time: {end - start}")
            total_time+=(end-start)

            gc.collect()
        
        if j%save_epoch==1:
            # 保存模型
            output_model_file = f'./models/model{j}/model.bin'
            output_vocab_file = f'./models/model{j}'
            if not os.path.exists(f'./models/model{j}'):
                os.makedirs(f'./models/model{j}')
            torch.save(model, output_model_file)
            tokenizer.save_pretrained(output_vocab_file)
            print('All files saved')
          
    print('\n\ntotal_training_time:',total_time)
    # generate test data
    #sample_data是测试集中前一千的数据
    test_data,test_labels_encoded,test_class,sample_data,test_labels = test_input_process(test_size=test_size,tokenizer=tokenizer)
    test_class_encoded = list(map(lambda x:ClassEncoder(x),test_class))
    test_dataset = processDataset(test_data, test_labels_encoded,test_class_encoded)
    testing_loader = DataLoader(test_dataset, **test_params)  
    
    # test process
    test(model=model,logging_step=logging_step,testing_loader=testing_loader,optimizer=optimizer,test_class_part=test_class[:1000],sample_data=sample_data)
    
    # 保存最终模型
    output_model_file = './models/model_final/model.bin'
    output_vocab_file = './models/model_final'
    if not os.path.exists('./models/model_final'):
        os.makedirs('./models/model_final')
    torch.save(model, output_model_file)
    tokenizer.save_pretrained(output_vocab_file)
    print('All files saved')


# In[24]:


if __name__=="__main__":
    #注意最开始的label处理部分，已经把没意义的赋值为-100了
    cap_attrs = {'Capacitance':1,'SizeCode':2,'RatedDCVoltageURdc':3,'PositiveTolerance':4,'NegativeTolerance':5,'TemperatureCharacteristicsCode':6,'MfrPartNumber':7,'input class':8}
    res_attrs = {'Resistance':11,'SizeCode':12,'WorkingVoltage':13,'Tolerance':14,'RatedPowerDissipationP':15,'MfrPartNumber':16,'input class':17}
    all_class_list = {'Capacitors':cap_attrs,'Resistors':res_attrs}

    cap_attrs_dict = dict(zip(cap_attrs.values(), cap_attrs.keys()))
    res_attrs_dict = dict(zip(res_attrs.values(), res_attrs.keys()))
    all_attrs_dict ={**cap_attrs_dict,**res_attrs_dict}
    
    write_result = True
    set_seed(1024)
    model_checkpoint = "albert-base-v2"
    #model_checkpoint = "prajjwal1/bert-tiny"
    # model_checkpoint = r'C:\Users\coldkiller\Desktop\supplyframe\checkpoint-3500'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    print('torch.cuda.is_available()',device)
    config = {'num_labels1':9,'num_labels2':8,'model':"prajjwal1/bert-tiny"}
    
    # Creating the loss function and optimizer
    LEARNING_RATE = 1e-5
    TRAIN_BATCH_SIZE=32
    VALID_BATCH_SIZE=32
    NUM_WORKERS=6
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)

    main(num_epoch=100,save_epoch=10,logging_step=100,file_num=10,train_size=100000,test_size=100000)


# In[ ]:


#train(num_epoch=2,logging_steps=1,file_num=30,train_size=100,test_size=300)


# In[ ]:


#训练和测试数据数量，epoch,dataloader_num_workers,模型,sample的数量


# In[ ]:


# read real data
#     with open(r'preprocess/description_with_label.json', 'r', errors='ignore', encoding='utf-8') as f:
#         js = f.read()
#         real_data = json.loads(js, strict=False)

#     real_input=[]
#     real_labels=[]
#     for i in range(len(real_data)):
#         if 'class' in real_data[i]:
#             real_input.append(real_data[i]['description'])
#             real_labels.append(attrEncoder(real_data[i]['class']))

#     print('real_input',len(real_input))
#     real_encodings = tokenizer(real_input,padding=True,truncation=False)
#     real_dataset = processDataset(real_encodings,real_labels)


#     real_preds = [attrDecoder(item) for item in real_result.predictions.argmax(-1)]
#     real_labels = [attrDecoder(item) for item in real_labels]
#     real_res = pd.DataFrame({"description":real_input,"true_labels":real_labels,"predicted_labels":real_preds})
#     real_res.to_csv('real_description_result.csv',encoding='utf_8_sig')

#     with open('running_output.txt','a') as f:
#         f.write(f"training time: {end - start}"+'\n')
#         f.write('train_dataset'+str(train_result)+'\n')        
#         f.write('test_dataset'+str(test_result)+'\n')
    
#     with open('real_description_output.json','w',errors='ignore') as f: 
#         json.dump(class_result,f,ensure_ascii=False, indent = 4) 

