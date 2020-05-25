import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def torch_distanceMatrix(X, Y=None):
    if(X.ndim == 1):
        X = X.reshape(len(X), 1)
    if(Y is None):
        Y = X
    else:
        if(Y.ndim == 1):
            Y = Y.reshape(len(Y), 1)
    n = X.size(0)
    m = Y.size(0)
    Xe = X.unsqueeze(1).expand(n, m, 1)
    Ye = Y.unsqueeze(0).expand(n, m, 1)
    return torch.abs(Xe - Ye).sum(2)


def torch_distanceCov(X, Y, correlation=False):
    A = torch_distanceMatrix(X)
    B = torch_distanceMatrix(Y)
    Ameans = A.mean(dim=0)
    Bmeans = B.mean(dim=0)
    A = A-Ameans-Ameans.reshape((len(Ameans), 1)) + Ameans.mean()
    B = B-Bmeans-Bmeans.reshape((len(Bmeans), 1)) + Bmeans.mean()
    distCov = torch.sqrt((A*B).mean())
    if(correlation):
        varA = torch.sqrt((A*A).mean())
        varB = torch.sqrt((B*B).mean())
        distCov = torch.div(distCov, torch.sqrt(varA * varB))

    return distCov


# def torch_distanceVar(X):
#     return torch_distanceCov(X, X)


def torch_cov(m, rowvar=False, inplace=False, correlation=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    covmatrix = fact * m.matmul(mt).squeeze()
    idxs = np.diag_indices(covmatrix.shape[0])
    V = covmatrix[idxs[0], idxs[1]]
    n = len(V)
    k = 0
    C = torch.empty(n*(n-1)//2, dtype=torch.float32).cuda()
    for i in range(n):
        for j in range(i):
            t = torch.div(covmatrix[i][j], torch.sqrt(V[i]*V[j]))
            C[k] = torch.pow(t, 2)
            k += 1

    return C.mean()


def calculateTripletLoss(embeddings, triplets, margin):
    pivot = embeddings[triplets[:, 0]]
    positives = embeddings[triplets[:, 1]]
    negatives = embeddings[triplets[:, 2]]
    ap_distances = (pivot - positives).pow(2).sum(1)  # .pow(.5)
    an_distances = (pivot - negatives).pow(2).sum(1)  # .pow(.5)
    triplet_loss = F.relu(ap_distances - an_distances + margin)
    return triplet_loss.mean()


class CorrelationMatrixLoss(nn.Module):
    LOG_FILE = open('logfile_lincorr.txt', 'w')

    def __init__(self, margin, triplet_selector):
        super().__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.alfa = 0.1

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        corr_loss = torch_cov(embeddings, correlation=True)
        # corr_loss = covmatrix.sum()

        triplet_loss = calculateTripletLoss(embeddings, triplets, self.margin)
        CorrelationMatrixLoss.LOG_FILE.write("%f %f\n" % (corr_loss, triplet_loss))
        CorrelationMatrixLoss.LOG_FILE.flush()

        return triplet_loss+self.alfa * corr_loss, len(triplets)
        # return triplet_loss, len(triplets)


class DistanceCorrelationLoss(nn.Module):
    LOG_FILE = open('logfile.txt', 'w')

    def __init__(self, margin, triplet_selector):
        super().__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.alfa = 0.5

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        triplet_loss = calculateTripletLoss(embeddings, triplets, self.margin)

        n = embeddings.shape[1]
        k = 0
        C = torch.empty(n*(n-1)//2, dtype=torch.float32).cuda()
        for i in range(n):
            for j in range(i):
                v = torch_distanceCov(embeddings[:, i], embeddings[:, j], correlation=True)
                # C[k] = torch.pow(v, 2)
                C[k] = v
                k += 1

        corr_loss = C.mean()
        DistanceCorrelationLoss.LOG_FILE.write("%f %f\n" % (corr_loss, triplet_loss))
        DistanceCorrelationLoss.LOG_FILE.flush()

        return triplet_loss+corr_loss, len(triplets)


class MITripletLoss(nn.Module):
    def __init__(self, margin, triplet_selector):
        super().__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.alfa = 5

    @staticmethod
    def MI(z, zt):
        C = z.shape[1]
        P = (z.unsqueeze(2)*zt.unsqueeze(1)).sum(dim=0)
        P = ((P + P.t()) / 2) / P.sum()
        # P[(P < EPS).data] = EPS
        Pi = P.sum(dim=1).view(C, 1).expand(C, C)
        Pj = P.sum(dim=0).view(1, C).expand(C, C)
        return (P*(torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        pivot = embeddings[triplets[:, 0]]
        positives = embeddings[triplets[:, 1]]
        negatives = embeddings[triplets[:, 2]]

        triplet_loss = calculateTripletLoss(pivot, positives, negatives)

        # mutual information
        positives_sm = F.softmax(positives[:, :3], dim=1)
        negatives_sm = F.softmax(negatives, dim=1)
        pivot_sm = F.softmax(positives[:, 3:], dim=1)
        mi = MyCustomTripletLoss.MI(positives_sm, pivot_sm)
        # print("===========")
        # print(iic.shape)
        # print(iic)
        # print('============')

        return triplet_loss+self.alfa*mi, len(triplets)
