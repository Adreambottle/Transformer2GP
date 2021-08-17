# README
    

### 1. 指示说明
* 项目路径位于 `/data/bob/Daniel_STF/trans2gp/`

* 因为对 `transformers` 和 `simpletransformer` 都进行了修改，所以以本地加载 library 的形式

* `transformers` 修改为 `adj_tf`，修改后位于 `/data/bob/Daniel_STF/trans2gp/adj_tf/`

* `simpletransformers` 修改为 `adj_stf`，修改后位于 `/data/bob/Daniel_STF/trans2gp/adj_stf/`

***

### 2.执行流程

#### 2.1 调用逻辑
* 执行 `main()` 函数

* `main()` 函数会调用 `./adj_stf.ner.ner_model.py` 文件的 `NERModel` 类

* `NERModel` 调用 `./adj_stf.ner.DNN2GP.py` 文件

* `NERModel` 调用 `./adj_tf/models/bert/modeling_bert.py` 文件


#### 2.2 在`main()`中传入的参数
* `MODEL_PATH = "./models/04-23_bert-base-cased/"` fine-tune 的模型地址 

* `CANDIDATE_NUM = 5` 返回出的最多的 candidate label 的上限

* `VAR_BOUND = 0.001` variance 的 upper bound

* `INPUT_DATA` 是一个 list，每条数据是一个字符串，不同的描述使用 `space` 隔开

```
INPUT_DATA = ['Resistors 155℃ P11P4F0GGSA20104MA 6669O 5% -400,400ppm/℃ 1W 2512',
              '4618pF 3848V X7R',
               ...
              '电阻 1206 100ppm/℃ 0.24 56.2Kohm']
```


#### 2.3 计算逻辑

* 通过 `simpletransformers` 的 `NERModel` 类对input data 进行预测

* 通过 `DNN2GP` 中的函数对 `BertForTokenClassification` 的最后一层进行 GP 计算，返回 variance

* 提取有最大 mean 的 `CANDIDATE_NUM` 个 labels 作为潜在的 candidate label

* 判断潜在的 `CANDIDATE_NUM` 个 candidate label 的 variance 是否大于 `VAR_BOUND`

* 如果出现 candidate label 的 variance 大于 `VAR_BOUND`，则模型对于该 word 分配到此 label 上的判断不是很确信，则输出此 label 作为该 word 的 candidate label

* 如果出现 candidate label 的 variance 小于 `VAR_BOUND`，则模型对于该 word 分配到此 label 上的判断比较确信，则循环停止

 

#### 2.3 返还的结果
* 返回的结果储存在 `outputs_data_total` 中

* Example: `INPUT_DATA = ["SURFACE MOUNT"]`

* 会返回每条 data 的每个 word 的预测的 label 和 candidate label
    
    * `print(outputs_data_total)`

    ```
    words: SURFACE
    labels: MountingFeature
    candidate:
        label:MountingFeature
        index:31
        mean:15.962848663330078
        variance:4.8210861081088296e-08
        
    words: MOUNT
    labels: others
    candidate:
        label:others
        index:38
        mean:6.4925737380981445
        variance:0.03210271894931793
        
        label:Capacitance
        index:11
        mean:2.2342257499694824
        variance:0.003786933608353138
        
        label:ESR
        index:118
        mean:1.742205262184143
        variance:0.7892318964004517
    ```


### 2. 在 Transformer 中修改的部分

* 修改文件的路径：`./adj_tf/models/bert/modeling_bert.py`

* 修改文件的部分： `class BertForTokenClassification(BertPreTrainedModel):`

* 修改的内容：


*  在 `BertForTokenClassification(BertPreTrainedModel)` 类下面添加了 `is_pred` 成员变量；

    * 在 `SimpleTransformer` 的 `NERModel()` 调用 `NERModel.predict()` 的时候，`is_pred = True`；
    * 在 `is_pred = False` 的情况下，`BertForTokenClassification.forward()` 的输出不会改变
* 在 `is_pred = False` 的情况下，对 `BertForTokenClassification.forward() `的输出进行修改：

    * 在输出中增加了 `sequence_output` 部分，即数据流在经过最后一层线性层的之前状态。  

***

### 3. 在 Simpletransformer 中修改的部分

* 添加的文件路径：在 `./adj_stf/ner/` 文件夹中添加 `DNN2GP.py` 文件

* 修改的文件路径：`./adj_stf/ner/ner_model.py`

* 修改的部分：`class NERModel` 的 `predict()`

* 修改的内容：

#### 3.1 在导入文件时添加 `DNN2GP` 中的函数
```python 
from adj_stf.ner.DNN2GP import (compute_dnn2gp_quantities,
                               compute_laplace,
                               token_length_and_first_token_id)
```
   

#### 3.2 在 `predict()` 的传入参数与输出

* 输入的参数增加了 `prior_prec`, `gp_batch_size`, `candidate_num` 3 个在 `main()` 中定义的参数

* 返回的 `preds` 是对每个 word 的预测的 label

* 返还的 `word_outputs` 包括了三部分
    * `word_top_k_index`      有最大 mean 的 K 个 label 对应的 index
    
    * `word_mean`             有最大 mean 的 K 个 label 对应的 mean
    
    * `word_variance`         有最大 mean 的 K 个 label 对应的 variance


```python
def predict(self, to_predict, split_on_space=True, prior_prec=1, gp_batch_size=2, var_num=5):

        ...
        
    return preds, word_outputs
```


#### 3.3 具体修改的部分


```python

"""
GP的过程
在这里进行了修改
"""
token_id_dict = token_length_and_first_token_id(out_label_ids, out_input_ids)             # 获取每条 data 的每个 word 的 first_token 对应的位置
before_last_layer_total = before_last_layer_total.clone().detach()                        # 获取最后一层线性层之前的数据，并使数据脱离计算图
gp_loader = DataLoader(before_last_layer_total, batch_size=gp_batch_size, shuffle=True)   # 新建用于 GP 的 data_loader
model_last_layer = model.classifier                                                       # 提取 BertForTokenClassification 的最后一层线性层

post_prec = compute_laplace(model=model_last_layer,                        # 获取后验分布的 variance
                            train_loader=gp_loader,                        # compute_laplace 来源于 DNN2GP
                            prior_prec=prior_prec,
                            device=self.device)

variances = compute_dnn2gp_quantities(model=model_last_layer,              # 计算模型需要输出的 variance
                          data_loader=gp_loader,                           # compute_dnn2gp_quantities 来源于 DNN2GP
                          device=self.device,
                          post_prec=post_prec)


word_mean = []
word_variance = []
word_top_k_index = []
for i in range(len(variances)):                                             # i 代表第 i 条 input_data
    variance = variances[i]                                                 # variance 是每条 data 对应的 variance  (128, 120)
    mean = token_logits[i, :, :]                                            # mean 是每条 data 对应的 mean          (128, 120)
    first_token_id_total = token_id_dict["first_token_id_total"][i]         # first tokens 的 id                  (words_number)
    first_token_variance = variance[first_token_id_total, :]                # first tokens 对应的 variance         (words_number, 120)
    first_token_mean = mean[first_token_id_total, :]                        # first tokens 对应的 mean             (words_number, 120)

    top_k_index = np.argsort(-first_token_mean, axis=1)[:,:var_num]         # 找到前 K 个有最大 mean 的 label id     (words_num, K)

    # 新建 word_mean_top_k, word_variance_top_k 两个新的矩阵，维度和 top_k_index 一样的
    # 将 top_k_index 对应的 mean 和 variance 储存到两个新的矩阵中
    word_mean_top_k = np.empty(top_k_index.shape)                           # 将这 K 个 label 对应的 mean 储存到新的矩阵里
    word_variance_top_k = np.empty(top_k_index.shape)                       # 将这 K 个 label 对应的 variance 储存到新的矩阵里
    for n in range(len(first_token_id_total)):
        for m in range(var_num):
            word_mean_top_k[n, m] = first_token_mean[n, top_k_index[n, m]]           # 前 K 个最大 mean 的 label 对应的 mean
            word_variance_top_k[n, m] = first_token_variance[n, top_k_index[n, m]]   # 前 K 个最大 mean 的 label 对应的 variance

    # 只保留前 K 个最大 mean 的 label 对应的 mean, variance 和 label_id
    word_top_k_index.append(top_k_index)                   # shape: (data_num [word_num, var_num])
    word_mean.append(word_mean_top_k)                      # shape: (data_num [word_num, var_num])
    word_variance.append(word_variance_top_k)              # shape: (data_num [word_num, var_num])

    word_outputs = {"word_top_k_index":word_top_k_index,       # 返回的是一个字典
                    "word_mean":word_mean,
                    "word_variance":word_variance}

return preds, word_outputs
```
   
