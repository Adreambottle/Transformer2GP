import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)



def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    """
    对应的是 nerClass
    """
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    """
    这里是设定setup
    """
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)




def run_demo(demo_fn, world_size):
    """需要用spawn并行"""
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


for para in paras:
    mp.spawn(fun, args=(para,), nprocs=2, join=True)

# def demo_model_parallel(rank, world_size):
#     print(f"Running DDP with model parallel example on rank {rank}.")
#     setup(rank, world_size)
#
#     # setup mp_model and devices for this process
#     dev0 = rank * 2
#     dev1 = rank * 2 + 1
#     mp_model = ToyMpModel(dev0, dev1)
#     ddp_mp_model = DDP(mp_model)
#
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)
#
#     optimizer.zero_grad()
#     # outputs will be on dev1
#     outputs = ddp_mp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(dev1)
#     loss_fn(outputs, labels).backward()
#     optimizer.step()
#
#     cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 8:
      print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")
    else:
      run_demo(demo_basic, 8)
      # run_demo(demo_checkpoint, 8)
      # run_demo(demo_model_parallel, 4)



















# from https://zhuanlan.zhihu.com/p/74792767
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os

input_size = 5
output_size = 2
batch_size = 30
data_size = 30

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("In Model: input size", input.size(),
              "output size", output.size())
        return output
model = Model(input_size, output_size)

if torch.cuda.is_available():
    model.cuda()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 就这一行
    model = nn.DataParallel(model)

for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)
    output = model(input_var)
    print("Outside: input size", input_var.size(), "output_size", output.size())


torch.distributed.init_process_group(backend="nccl")
model=torch.nn.parallel.DistributedDataParallel(model)



def out(fun, **kwargs):
    fun()