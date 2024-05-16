from torch.nn.modules import loss
import torch
import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae,mse,rmse,mape,mspe

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

class TopkMSELoss(torch.nn.Module):
    #이건 Long-Rang Forecasting에서 사용되는 Loss
    def __init__(self, topk) -> None:
        super().__init__()
        self.topk = topk
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, output, label):
        losses = self.criterion(output, label).mean(2).mean(1)
        losses = torch.topk(losses, self.topk)[0]

        return losses

class SingleStepLoss(torch.nn.Module):
    """ Compute top-k log-likelihood and mse. """
    #0인 값은 loss 계산할 때 빼기 위해 ignore_zero라는 변수를 사용
    def __init__(self, ignore_zero):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(self, mu, sigma, labels, topk=0):
        if self.ignore_zero:
            indexes = (labels != 0)
        else:
            indexes = (labels >= 0)

        #mu, sigma는 예측치의 평균과 표준편차
        #이때 likelihood의 길이는 batch의 길이와 같음
        distribution = torch.distributions.normal.Normal(mu[indexes], sigma[indexes])
        likelihood = -distribution.log_prob(labels[indexes])
        #Squared Error 계산
        diff = labels[indexes] - mu[indexes]
        se = diff * diff

        #top-k loss를 계산하기 위해 topk가 0보다 크면 top-k개의 loss만 반환해줌
        #[0]이 있어서 헷갈릴 수 있는데 k개 반환해주는거 맞음
        #topk가 0이면 이 부분이 진행되지 않으므로 전체 loss를 반환해줌
        if 0 < topk < len(likelihood):
            likelihood = torch.topk(likelihood, topk)[0]
            se = torch.topk(se, topk)[0]

        #tok-k log-likelihood 와 tok-k SE를 반환해줌
        #이 부분은 single step main에서 criterion이라는 이름으로 사용되고 있는데
        #그 과정이 조금 복잡하긴 하지만 아무튼 log-likelihood와 SE를 반환해주는 함수
        #주의사항 : MSE가 아니라 SE를 반환해주는 것
        return likelihood, se

def AE_loss(mu, labels, ignore_zero):
    #이건 평범한 AE 반환
    if ignore_zero:
        indexes = (labels != 0)
    else:
        indexes = (labels >= 0)

    ae = torch.abs(labels[indexes] - mu[indexes])
    return ae
