import json
import random
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
os.chdir("/data/bob/Daniel_STF/trans2gp")
sys.path.append("..")
sys.path.append("/data/bob/Daniel_STF/trans2gp")
from adj_stf.ner.ner_model import NERModel, NERArgs



def main(model_path=None, input_data=None, candidate_num=5, var_bound=0.001):

    # load label list
    print("The model is:", model_path.split('/')[-2])
    with open(model_path + 'model_args.json', 'r') as file:
        model_args = json.loads(file.read())
        labels_list = model_args['labels_list']

    model = NERModel('bert', model_path, use_cuda=False, labels=labels_list, args=model_args)

    # get results of model.predict
    preds, word_outputs = model.predict(input_data, prior_prec=1, gp_batch_size=4, candidate_num=candidate_num)

    # get variance
    label_map = {i: label for i, label in enumerate(labels_list)}


    word_top_k_index = word_outputs["word_top_k_index"]           # shape: (data_num, [word_num, K])
    word_mean = word_outputs["word_mean"]                         # shape: (data_num, [word_num, K])
    word_variance = word_outputs["word_variance"]                 # shape: (data_num, [word_num, K])


    outputs_data_total= []
    candidate_total = []
    for i, data in enumerate(preds):                # 所有 input_data 中的第 i 条 data

        mean = word_mean[i]                         # 获取第 i 条 data 中 top_k 个 的 mean, variance 和 index
        variance = word_variance[i]
        top_k_index = word_top_k_index[i]

        outputs_data_list = []
        candidate_word = []
        for j, word in enumerate(data):             # 每条 data 中第 j 个 word

            candidate_list = []
            for k in range(mean.shape[1]):          # 每个 word 对应的最大的 K 个labels

                # 将第 j 个 word 的 top_k 个 label 上的 variance 和 mean 输出
                candidate_dict = {"label": label_map[top_k_index[j, k]],
                                  "index": top_k_index[j, k],
                                  "mean": mean[j, k],
                                  "variance": variance[j, k]}
                candidate_list.append(candidate_dict)

                # 如果第出现 variance 小于 VAR_BOUND 的情况，则停止循环
                if variance[j, k] < var_bound:
                    break

            outputs_data = {"words":list(word.keys())[0],
                            "labels":list(word.values())[0],
                            "candidate":candidate_list}
            outputs_data_list.append(outputs_data)
            candidate_word.append(candidate_list)
        outputs_data_total.append(outputs_data_list)
        candidate_total.append(candidate_word)

    for i, outputs in enumerate(outputs_data_total):
        print(f"\ndata{i}:")
        for item in outputs:
            print("\twords:",item["words"])
            print("\tlabels:",item["labels"])
            print("\tcandidate:",)
            for candidate in item["candidate"]:
                for key, value in candidate.items():
                    print(f"\t\t{key}:{value}")
                print("\n")
            print("\n")






if __name__ == '__main__':

    MODEL_PATH = "./models/04-23_bert-base-cased/"
    PRED_DATA_PATH = '/data/bob/synthetic/2021-04-22_4m/1m_synthetic_cap.json'
    PRED_DATA_LEN = 1
    CANDIDATE_NUM = 5
    VAR_BOUND = 0.001

    INPUT_DATA = [
        'Resistors 155℃ P11P4F0GGSA20104MA 6669O 5% -400,400ppm/℃ 1W 2512',
        '4618pF 3848V X7R',
        '电容 ALUMINUM 105℃ 9840pF 20% 100V',
        '255Kohm 0.59 1010 150℃',
        '电阻 -55℃ CRCW0805P1134FRT5 5313G 0.36 SMT 2/5W 1206',
        '-10.10ppm 4011Ω',
        '电阻 1206 100ppm/℃ 0.24 56.2Kohm',
        'cap 1253u 63000m -40℃ C2216 SURFACE MOUNT',
    ]


    main(model_path=MODEL_PATH, input_data=INPUT_DATA, candidate_num=CANDIDATE_NUM, var_bound=VAR_BOUND)
