import torch
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn

# from dnn2gp.neural_networks import LeNet5, LeNet5CIFAR
# from dnn2gp.datasets import Dataset
# from dnn2gp import compute_laplace, compute_dnn2gp_quantities, compute_kernel

torch.set_default_dtype(torch.double)
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(77)
np.random.seed(77)

class LeNet5(nn.Module):
    def __init__(self, input_channels=1, dims=28, num_classes=2):
        super(type(self), self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.fc1 = nn.Linear(16*dims*dims, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc(out)
        return out


def gradient(model):
    grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
    return grad.detach()


def weights(model):
    # 返还所有的参数
    wts = torch.cat([p.flatten() for p in model.parameters()])
    return wts


def compute_kernel(Jacobians, agg_type='diag'):
    """
    用来计算相似度矩阵的
    Compute kernel by various aggregation types based on Jacobians
    """
    if agg_type == 'diag':
        K = np.einsum('ikp,jkp->ij', Jacobians, Jacobians)  # one gp per class and then sum
    elif agg_type == 'sum':
        K = np.einsum('ikp,jlp->ij', Jacobians, Jacobians)  # sum kernel up
    elif agg_type == 'full':
        K = np.einsum('ikp,jlp->ijkl', Jacobians, Jacobians)  # full kernel NxNxKxK
    else:
        raise ValueError('agg_type not available')
    return K

def compute_laplace_old(model, train_loader, prior_prec, device):
    """
    用来计算后验概率的
    Compute diagonal posterior precision due to Laplace approximation
    :param model: pytorch neural network
    :param train_loader: data iterator of training set with features and labels
    :param prior_prec: prior precision scalar
    :param device: device to compute/backpropagate on (ideally GPU)
    """
    theta_star = weights(model)
    post_prec = (torch.ones_like(theta_star) * prior_prec)
    for data, label in tqdm(train_loader):
    # for batch in tqdm(train_loader, disable=args.silent, desc="Running Prediction"):
    #    batch = tuple(t.to(device) for t in batch)
        data, label = data.to(device).double(), label.to(device).double()
        prediction = model.forward(data)
        p = torch.softmax(prediction, -1).detach()
        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        Jacs = list()
        for i in range(prediction.shape[0]):
            Jac = list()
            for j in range(prediction.shape[1]):
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jij = gradient(model)
                Jac.append(Jij)
                model.zero_grad()
            Jac = torch.stack(Jac).t()
            Jacs.append(Jac)
        Jacs = torch.stack(Jacs).detach()
        post_prec += torch.einsum('npj,nij,npi->p', Jacs, Lams, Jacs)
    return post_prec

def compute_laplace(model, train_loader, prior_prec, device):
    """
    用来计算后验概率的
    Compute diagonal posterior precision due to Laplace approximation
    :param model: pytorch neural network
    :param train_loader: data iterator of training set with features and labels
    :param prior_prec: prior precision scalar
    :param device: device to compute/backpropagate on (ideally GPU)
    """
    theta_star = weights(model)
    post_prec = (torch.ones_like(theta_star) * prior_prec)
    for data in tqdm(train_loader):
        # print("Here", data.shape)
        # print(data)
        data = data.to(device).double()
        prediction = model.forward(data[0])
        p = torch.softmax(prediction, -1).detach()
        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        Jacs = list()
        for i in range(prediction.shape[0]):
            Jac = list()
            for j in range(prediction.shape[1]):
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jij = gradient(model)
                Jac.append(Jij)
                model.zero_grad()
            Jac = torch.stack(Jac).t()
            Jacs.append(Jac)
        Jacs = torch.stack(Jacs).detach()
        post_prec += torch.einsum('npj,nij,npi->p', Jacs, Lams, Jacs)
    return post_prec

def compute_dnn2gp_quantities_old(model, data_loader, device, limit=-1, post_prec=None):
    """
    用来得出一堆参数的
    Compute reparameterized nn2gp quantities for softmax regression (multiclassification)
    :param model: pytorch function subclass with differentiable output
    :param data_loader: data iterator yielding tuples of features and labels
    :param device: device to do heavy compute on (saving and aggregation on CPU)
    :param limit: maximum number of data to iterate over
    :param post_prec: posterior precision (diagonal)
    """
    Jacobians = list()
    predictive_mean_GP = list()
    labels = list()
    predictive_var_f = list()
    predictive_noise = list()
    predictive_mean = list()
    theta_star = weights(model)
    n_points = 0
    for data, label in tqdm(data_loader):
        data, label = data.to(device).double(), label.to(device).double()
        prediction = model.forward(data)
        p = torch.softmax(prediction, -1).detach()

        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        y_uct = p - (p ** 2)
        for i in range(prediction.shape[0]):
            Jacs = list()
            kpreds = list()
            for j in range(prediction.shape[1]):
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jij = gradient(model)
                Jacs.append(Jij)
                with torch.no_grad():
                    kpreds.append(Jij @ theta_star)
                model.zero_grad()
            Jacs = torch.stack(Jacs)
            Jacobians.append(Jacs.to('cpu'))
            jtheta_star = torch.stack(kpreds).flatten()
            predictive_mean_GP.append(jtheta_star.to('cpu'))
            predictive_mean.append(p[i].to('cpu'))
            if post_prec is not None:
                f_uct = torch.diag(Lams[i] @ torch.einsum('kp,p,mp->km', Jacs, 1/post_prec, Jacs)
                                   @ Lams[i])
                predictive_var_f.append(f_uct.to('cpu'))
                predictive_noise.append(y_uct[i].to('cpu'))

        labels.append(label.to('cpu'))
        n_points += data_loader.batch_size
        if n_points >= limit > 0:
            break

    if post_prec is not None:
        return (torch.stack(Jacobians),
                torch.stack(predictive_mean_GP),
                torch.stack(labels).flatten(),
                torch.stack(predictive_var_f),
                torch.stack(predictive_noise),
                torch.stack(predictive_mean))
    return torch.stack(Jacobians), torch.stack(predictive_mean_GP), torch.stack(labels).flatten()

def compute_dnn2gp_quantities(model, data_loader, device, limit=-1, post_prec=None):
    """
    用来得出一堆参数的
    Compute reparameterized nn2gp quantities for softmax regression (multiclassification)
    :param model: pytorch function subclass with differentiable output
    :param data_loader: data iterator yielding tuples of features and labels
    :param device: device to do heavy compute on (saving and aggregation on CPU)
    :param limit: maximum number of data to iterate over
    :param post_prec: posterior precision (diagonal)
    """
    Jacobians = list()
    predictive_mean_GP = list()
    labels = list()
    predictive_var_f = list()
    predictive_noise = list()
    predictive_mean = list()
    theta_star = weights(model)
    n_points = 0
    count = 0
    for data in tqdm(data_loader):


        data = data.to(device).double()
        with open("./results/datastream.log", "a+") as f:
            print("\ncount:", count, file=f)
            print("\ndata shape\n", data.shape, file=f)
            print("\ndata\n", data, file=f)

        prediction = model.forward(data[0])
        p = torch.softmax(prediction, -1).detach()

        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        with open("./results/lams.log", "a+") as f:
            print("\ncount:", count, file=f)
            print("\n lamda shape\n", Lams.shape, file=f)
            print("\n lamda\n", Lams, file=f)

        y_uct = p - (p ** 2)
        for i in range(prediction.shape[0]):
            print("i:", i)
            Jacs = list()
            kpreds = list()
            for j in range(prediction.shape[1]):
                # print("j:", j)
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jij = gradient(model)
                Jacs.append(Jij)
                with torch.no_grad():
                    kpreds.append(Jij @ theta_star)
                model.zero_grad()
            Jacs = torch.stack(Jacs)
            Jacobians.append(Jacs.to('cpu'))
            jtheta_star = torch.stack(kpreds).flatten()
            predictive_mean_GP.append(jtheta_star.to('cpu'))
            predictive_mean.append(p[i].to('cpu'))
            if post_prec is not None:
                f_uct = torch.diag(Lams[i] @ torch.einsum('kp,p,mp->km', Jacs, 1/post_prec, Jacs)
                                   @ Lams[i])
                predictive_var_f.append(f_uct.to('cpu'))
                predictive_noise.append(y_uct[i].to('cpu'))

        # labels.append(label.to('cpu'))
        n_points += data_loader.batch_size
        print("n_points ", n_points)
        count += 1
        if n_points >= limit > 0:
            break

    if post_prec is not None:
        return (torch.stack(Jacobians),
                torch.stack(predictive_mean_GP),
                # torch.stack(labels).flatten(),
                torch.stack(predictive_var_f),
                torch.stack(predictive_noise),
                torch.stack(predictive_mean))
    return torch.stack(Jacobians), torch.stack(predictive_mean_GP), torch.stack(labels).flatten()


def compute_kernel_and_predictive_laplace(model, loader, delta, fname):

    post_prec = compute_laplace(model, loader, delta, device)

    # 通过 compute_dnn2gp_quantities 计算
    Jacobians, predictive_mean_GP, labels, predictive_var_f, predictive_noise, predictive_mean = \
        compute_dnn2gp_quantities(model, loader, device, limit=1000, post_prec=post_prec)
    labels = labels.numpy()
    strat_labels = list()
    for i in range(10):
        ixs = np.where(labels == i)[0]
        strat_labels.append(ixs[:30])
    strat_labels = np.hstack(strat_labels)
    Jacobians = Jacobians.numpy()[strat_labels]
    predictive_var_f = predictive_var_f.numpy()[strat_labels]
    predictive_mean_GP = predictive_mean_GP.numpy()[strat_labels]
    predictive_noise = predictive_noise.numpy()[strat_labels]
    predictive_mean = predictive_mean.numpy()[strat_labels]
    K = compute_kernel(Jacobians, agg_type='diag')  # one gp per class and sum variances

    np.save('results/{fname}_Laplace_gp_predictive_mean'.format(fname=fname), predictive_mean_GP)
    np.save('results/{fname}_Laplace_predictive_mean'.format(fname=fname), predictive_mean)
    np.save('results/{fname}_Laplace_predictive_var_f'.format(fname=fname), predictive_var_f)
    np.save('results/{fname}_Laplace_predictive_noise'.format(fname=fname), predictive_noise)
    np.save('results/{fname}_Laplace_kernel'.format(fname=fname), K)


if __name__ == '__main__':
    ### MNIST
    transformations = transforms.Compose([transforms.ToTensor(), lambda x: x.double()])
    trainset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transformations)
    loader = DataLoader(trainset, batch_size=128, shuffle=True)

    model = LeNet5(num_classes=2).to(device)
    model.load_state_dict(torch.load('models/2_class_mnist_zone_lenet_vogn.tk', map_location=device))
    # compute_kernel_predictive_VI(model, loader, 'BIN_MNIST')

    model = LeNet5(num_classes=10).to(device)
    model_last_layer = model.fc
    # Adam with weight-decay = 1e-4
    model.load_state_dict(torch.load('models/full_mnist_lenet_adaml2.tk', map_location=device))
    compute_kernel_and_predictive_laplace(model, loader, 1e-4, 'MNIST')
    # for VOGN model, prior precision = 1e-4
    model.load_state_dict(torch.load('models/full_mnist_lenet_vogn.tk', map_location=device))
    # compute_kernel_predictive_VI(model, loader, 'MNIST')


# for data, label in loader:
#     print(data.shape)
#     print(data)
#     print(label.shape)
#     print(label)
#     break