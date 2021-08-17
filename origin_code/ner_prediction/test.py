import torch
import numpy as np
a = torch.ones(1000)
b = torch.ones((10, 10, 10))

x = torch.Tensor([[1, -1], [-1, 1]])
y = torch.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
x.requires_grad_ = True
x.requires_grad = True
y.requires_grad_ = True
y.requires_grad = True

x.sum().item()
y.requires_grad = True

y.sum().item()

output = y.pow(2) + y
output = output.sum()
output.backward()


x
z = torch.pow(x, 2)
out = z.sum()
out.backward()
z_p = z.clone()
x_p = x.clone()

a = np.array([[1, 2, 3], [4, 5, 6]])
b = torch.Tensor().new_tensor(a)
c = torch.Tensor().new_full(a.shape, 4)
c = b.new_tensor(a)
a = a + 1

d = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)


import json

ds = {"a":1,
      "b":2,
      "c":
          {"c1":12,
           "c2":13}
      }
with open("config.json", 'w') as f:
    f.write(json.dumps(con))

con = {
  "train":
    {
      "num_labels1": 9,
      "num_labels2": 8,
      "model":
      {
        "albert": "albert-base-v2",
        "tiny": "prajjwal1/bert-tiny"
      }
    },
  "address":
    {
      "model_addr":
      {
        "output_model_file": "./models/model_final/model.bin",
        "output_vocab_file": "./models/model_final"
      },
      "input_addr": "./inputData/",
      "raw_path":
      {
        "on_server":
        {
          "cap_path": "/data/bob/synthetic/cap/",
          "res_path": "/data/bob/synthetic/res/"
        },
        "on_local":
        {
          "cap_path": "/data/bob/synthetic/cap/",
          "res_path": "/data/bob/synthetic/res/"
        }
      }
    },
  "Hyperparameter":
    {
      "LEARNING_RATE": 1e-5,
      "TRAIN_BATCH_SIZE": 24,
      "VALID_BATCH_SIZE": 24,
      "NUM_WORKERS": 6
    },
  "Main_Parameter":
    {
      "num_epoch": 100,
      "save_epoch": 1,
      "logging_step": 100,
      "file_num": 100,
      "train_size": 10000,
      "test_size": 10000
    }
}


from sklearn.metrics import classification_report
# from seqeval.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
reports = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=target_names))
print(reports)

from tqdm import tqdm
from time import sleep
alist = list('letters')
bar = tqdm(alist)
for letter in bar:
    sleep(1)
    bar.set_description(f"Now get {letter}")

import torch
a = torch.Tensor([[1, 2], [3, 4]])
a.requires_grad = True
print(a)
print(a.data)
a.item()
b = a[0, 0]
b.item()