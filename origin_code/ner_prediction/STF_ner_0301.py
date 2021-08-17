import json
import random
import pandas as pd

from fractions import Fraction
from simpletransformers.ner import NERModel, NERArgs

attention_probs_dropout_prob_list = [0.1, 0.75, 0.5, 0.25, 0]

amount = 100000

with open('/data/bob/synthetic/capacitors/1m_synthetic_cap0.json', 'r') as file:
    data = json.loads(file.read())[:amount]

with open('/data/bob/synthetic/resistors/1m_synthetic_res0.json', 'r') as file:
    data.extend(json.loads(file.read())[:amount])

with open('/data/bob/synthetic/others/1m_synthetic_others0.json', 'r') as file:
    data.extend(json.loads(file.read())[:amount])

random.shuffle(data)

for i in range(5):
    output = []
    labels = set()
    for index in range(len(data)):
        for i in range(len(data[index]['description'])):
            output.append([index, str(data[index]['description'][i]), str(data[index]['labels'][i])])
            labels.add(data[index]['labels'][i])
    train_data = pd.DataFrame(output, columns=['sentence_id', 'words', 'labels'])

    model_args = NERArgs()
    model_args.labels_list = list(labels)
    model_args.num_train_epochs = 1
    model_args.save_steps = 500000
    model_args.overwrite_output_dir = True

    model = NERModel('bert', 'bert-base-cased', args=model_args)
    config_dict = model.config.to_dict()
    config_dict["attention_probs_dropout_prob"] = attention_probs_dropout_prob_list[i]

    model.train_model(train_data)
