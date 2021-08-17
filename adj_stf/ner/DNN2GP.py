import torch
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn


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

    batch_num = 0
    for data in tqdm(train_loader):
        # print("Here", data.shape)
        print("\n\n\nPerforming compute_laplace")
        print("batch_num", batch_num)

        batch_num += 1

        # data = data.to(device).double()
        data = data.to(device).float()

        prediction = model.forward(data[0])
        p = torch.softmax(prediction, -1).detach()
        Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        Jacs_in_label = list()
        for i in range(prediction.shape[0]):
            Jac = list()
            for j in range(prediction.shape[1]):
                rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                prediction[i, j].backward(retain_graph=rg)
                Jacs_in_token = gradient(model)
                Jac.append(Jacs_in_token)
                model.zero_grad()
            Jac = torch.stack(Jac).t()
            Jacs_in_label.append(Jac)
        Jacs_in_label = torch.stack(Jacs_in_label).detach()
        post_prec += torch.einsum('npj,nij,npi->p', Jacs_in_label, Lams, Jacs_in_label)
    return post_prec


def compute_dnn2gp_quantities(model, data_loader, device, post_prec=None):
    """
    用来得出一堆参数的
    Compute reparameterized nn2gp quantities for softmax regression (multiclassification)
    :param model: pytorch function subclass with differentiable output
    :param data_loader: data iterator yielding tuples of features and labels
    :param device: device to do heavy compute on (saving and aggregation on CPU)
    :param post_prec: posterior precision (diagonal)
    """

    batch_num = 0
    data_num = 0
    # 先对每个batch进行循环
    variance_list = []
    for batch in tqdm(data_loader):
        print("\n\n\nPerforming compute_dnn2gp_quantities of batch", batch_num)
        for n, data in enumerate(batch):
            print("caluculating_data_num", data_num)

            Jacobians = list()
            predictive_mean_GP = list()
            predictive_var_f = list()
            predictive_noise = list()
            predictive_mean = list()

            theta_star = weights(model)


            batch_num += 1

            data = data.to(device).float()
            # data = data.to(device).double()

            prediction = model.forward(data)
            p = torch.softmax(prediction, -1).detach()

            Lams = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
            y_uct = p - (p ** 2)
            # print("prediction_shape", prediction.shape)

            # i = 128 Tokens 的个数
            for i in range(prediction.shape[0]):
                Jacs_in_label = list()
                kpreds = list()

                # j = 120 Lables 的个数
                for j in range(prediction.shape[1]):
                    rg = (i != (prediction.shape[0] - 1) or j != (prediction.shape[1] - 1))
                    prediction[i, j].backward(retain_graph=rg)
                    Jacs_in_token = gradient(model)

                    Jacs_in_label.append(Jacs_in_token)
                    with torch.no_grad():
                        kpreds.append(Jacs_in_token @ theta_star)
                    model.zero_grad()
                    # 这里计算完每一个矩阵的

                # 拼合每个 label 上的 Jacs 拼好
                Jacs_in_label = torch.stack(Jacs_in_label)
                jtheta_star = torch.stack(kpreds).flatten()

                # 在 tokens 的维度上append到新的列表中
                Jacobians.append(Jacs_in_label.to('cpu'))
                predictive_mean_GP.append(jtheta_star.to('cpu'))
                predictive_mean.append(p[i].to('cpu'))


                if post_prec is not None:
                    f_uct = torch.diag(Lams[i]
                                       @ torch.einsum('kp,p,mp->km', Jacs_in_label, 1/post_prec, Jacs_in_label)
                                       @ Lams[i])
                    predictive_var_f.append(f_uct.to('cpu'))
                    predictive_noise.append(y_uct[i].to('cpu'))

            predictive_var_f = torch.stack(predictive_var_f).numpy()

            variance_list.append(predictive_var_f)
            data_num += 1

    return variance_list


def token_length_and_first_token_id(out_label_ids, out_input_ids):
    """
    获取每条数据的有效长度和第一个token的index
    :return: (token_length_total, first_token_id_total)
            token_length_total:<list> 每条数据的长度
            first_token_id_total:<list> 每条数据第一个token的index
    """
    # out_label_ids = np.load(saving_path + '/out_label_ids.npy')
    # out_input_ids = np.load(saving_path + '/out_input_ids.npy')

    first_token_id_total = []
    token_length_total = []
    for i in range(out_label_ids.shape[0]):
        first_token_id = []
        total_length = 0
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] == 0:
                first_token_id.append(j)

            if out_input_ids[i, j] == 0:
                total_length = j - 1
                break
        first_token_id_total.append(first_token_id)
        token_length_total.append(total_length)
    return {"token_length_total":token_length_total,
            "first_token_id_total":first_token_id_total}