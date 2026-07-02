import sys
import random
from typing import Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
try:
    from cvxopt import solvers, matrix
except ImportError:
    # cvxopt is only needed for the MK-MMD kernel-weight QP solver (not used by CALM).
    solvers = matrix = None
from sklearn.gaussian_process.kernels import RBF
from typing import Optional

min_var_est = 1e-8


# taken from https://github.com/MaterialsInformaticsDemo/MK-MMD/blob/main/code/MK_MMD.py
class MKMMD:
    def __init__(
        self,
        gamma_list=[
            2,
            1,
            1 / 2,
            1 / 4,
            1 / 8,
        ],
        kernel_num=5,
    ):
        """
        Our code is designed for educational purposes,
        and to make it easier to understand,
        we have implemented only the RBF (Radial Basis Function) kernel.

        This case focuses on solving the weights of kernels.
        The estimation of length scales is crucial in kernel-based models,
        For further details on the method (length scales), please visit the following link:
        [https://github.com/MaterialsInformaticsDemo/DAN/blob/main/code/MK_MMD.py].

        :param gamma_list: list of length scales for rbf kernels
        :param kernel_num: number of kernels in MK_MMD
        """
        if len(gamma_list) != kernel_num:
            print("please assign specific length scales for each rbf kernel")
        self.kernel_num = kernel_num
        kernel_list = []
        for i in range(kernel_num):
            kernel_list.append(RBF(gamma_list[i], "fixed"))
        self.kernel_list = kernel_list

    def predict(
        self,
        Xs,
        Xt,
    ):
        """
        :param Xs: ns * m_feature, source domain data
        :param Xt: nt * m_feature, target domain data

        return :
        the result of MK_MMD & weights of kernels
        """
        # cal weights for each rbf kernel
        # two rows above section 2.2 Empirical estimate of the MMD, asymptotic distribution, and test
        h_matrix = []  # 5 * 5
        for i in range(self.kernel_num):
            _, h_k_vector = funs(
                Xs, Xt, self.kernel_list[i], MMD=False, h_k_vector=True
            )
            h_matrix.append(h_k_vector)
        h_matrix = np.vstack(h_matrix)
        print("h matrix is calculated")

        # cal the covariance matrix of h_matrix
        # Eq.(7)
        Q_k = np.cov(h_matrix)
        # cal the weights of kernels, Eq.(11)
        # vector η_k, Eq.(2)
        η_k = []
        for k in range(self.kernel_num):
            MMD, _ = funs(Xs, Xt, self.kernel_list[k], MMD=True, h_k_vector=False)
            η_k.append(MMD)
        print("η_k is calculated")

        # solve the standard quadratic programming problem
        # see : https://github.com/Bin-Cao/KMMTransferRegressor/blob/main/KMMTR/KMM.py
        P = 2 * matrix(Q_k + 1e-5 * np.eye(self.kernel_num))  # λm = 1e-5
        # q = - η_k ， maximum η_k * beta in QB
        q = matrix(-np.array(η_k).reshape(-1, 1))
        G = matrix(-np.eye(self.kernel_num))
        # the summation of the beta is 1, Eq.(3), let's D = 1
        A = matrix(np.ones((1, self.kernel_num)))
        b = matrix(1.0)
        h = matrix(np.zeros((self.kernel_num, 1)))
        # P is 5 * 5
        # q is 5 * 1
        # G is 5 * 5
        # A is 1 * 5
        # b = 1, h = 5*1
        solvers.options["show_progress"] = False
        sol = solvers.qp(P, q, G, h, A, b)
        beta = sol["x"]
        print("the optimal weights are found")
        MK_MMD = np.array(η_k) @ np.array(beta)

        kernel = beta[0] * self.kernel_list[0]
        for k in range(self.kernel_num - 1):
            kernel += beta[k + 1] * self.kernel_list[k + 1]

        return MK_MMD, np.array(beta), kernel


def funs(Xs, Xt, kernel, MMD=True, h_k_vector=False):
    if MMD == True:
        # cal MMD for one rbf kernel
        # Eq.(1) in paper
        dim = np.array(Xs).ndim
        Xs = np.array(Xs).reshape(-1, dim)
        Xt = np.array(Xt).reshape(-1, dim)
        EXX_ = kernel(Xs, Xs)
        EYY_ = kernel(Xt, Xt)
        EYX_ = kernel(Xt, Xs)
        EXY_ = kernel(Xs, Xt)
        MMD = (
            np.array(EXX_).mean()
            + np.array(EYY_).mean()
            - np.array(EYX_).mean()
            - np.array(EXY_).mean()
        )
    else:
        MMD = None
        pass

    if h_k_vector == True:
        # cal vector h_k(x,x',y,y'), contains m**2*n**2 terms
        # between Eq.(1) and Eq.(2)
        # k(x, x') is the element of matrix EXX_
        # k(y, y') is the element of matrix EYY_
        # k(x, y') and k(x', y) are the element of matrix EXY_
        ns, nt = len(Xs), len(Xt)
        combin_ns = generate_combinations(ns)
        combin_nt = generate_combinations(nt)
        h_k_vector = []
        for x in range(len(combin_ns)):
            for y in range(len(combin_nt)):
                S_x = np.array(Xs[combin_ns[x][0]]).reshape(-1, 1)  # x
                S_x_ = np.array(Xs[combin_ns[x][1]]).reshape(-1, 1)  # x'
                T_x = np.array(Xt[combin_nt[y][0]]).reshape(-1, 1)  # y
                T_x_ = np.array(Xt[combin_nt[y][1]]).reshape(-1, 1)  # y'
                h_k = (
                    kernel(S_x, S_x_)
                    + kernel(T_x, T_x_)
                    - kernel(S_x, T_x_)
                    - kernel(S_x_, T_x)
                )
                h_k_vector.append(h_k[0][0])
        h_k_vector = np.array(h_k_vector)
    else:
        h_k_vector = None
        pass
    return MMD, h_k_vector


# Code here is taken from: https://github.com/OctoberChang/MMD-GAN/blob/master/mmd.py


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert X.size(0) == Y.size(0)
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = (
            Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m)
        )

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased
    )
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX**2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY**2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY**2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = (
            Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m)
        )

    var_est = (
        2.0
        / (m**2 * (m - 1.0) ** 2)
        * (
            2 * Kt_XX_sums.dot(Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4.0 * m - 6.0) / (m**3 * (m - 1.0) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0
        * (m - 2.0)
        / (m**3 * (m - 1.0) ** 2)
        * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m**3 * (m - 1.0) ** 2) * (K_XY_2_sum)
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0
        / (m**3 * (m - 1.0))
        * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0)
        )
    )
    return mmd2, var_est


def contrastive_loss(
    z_alpha,
    labels_alpha,
    z_beta,
    labels_beta,
    lambda_param=0.5,
    tau=0.07,
    eps=1e-8,
):
    """
    Supervised Contrastive Loss for unpaired cross-modal alignment.
    Formula: -log(sum_pos / (sum_pos + sum_neg))
    """
    if z_alpha.numel() == 0 or z_beta.numel() == 0:
        return torch.tensor(0.0, device=z_alpha.device)

    z_a = F.normalize(z_alpha, p=2, dim=1)
    z_b = F.normalize(z_beta, p=2, dim=1)

    sim = torch.mm(z_a, z_b.t()) / tau
    mask_pos = (labels_alpha.view(-1, 1) == labels_beta.view(1, -1)).float()
    mask_neg = (
        1.0 - mask_pos
    )  # explicitly separate negatives to avoid counting positives in the denominator

    # ----- Direction A -> B -----
    exp_sim = torch.exp(sim)

    # Numerator: sum of exp(sim) for positives
    pos_sum = (exp_sim * mask_pos).sum(dim=1)

    # Denominator: positives + negatives (negatives only come from different-class samples)
    neg_sum = (exp_sim * mask_neg).sum(dim=1)
    all_sum = pos_sum + neg_sum

    valid_ab = mask_pos.sum(dim=1) > 0
    if valid_ab.any():
        loss_ab = -torch.log(
            (pos_sum[valid_ab] + eps) / (all_sum[valid_ab] + eps)
        ).mean()
    else:
        loss_ab = torch.tensor(0.0, device=z_alpha.device)

    # ----- Direction B -> A -----
    exp_sim_ba = exp_sim.t()
    mask_pos_ba = mask_pos.t()
    mask_neg_ba = mask_neg.t()

    pos_sum_ba = (exp_sim_ba * mask_pos_ba).sum(dim=1)
    neg_sum_ba = (exp_sim_ba * mask_neg_ba).sum(dim=1)
    all_sum_ba = pos_sum_ba + neg_sum_ba

    valid_ba = mask_pos_ba.sum(dim=1) > 0
    if valid_ba.any():
        loss_ba = -torch.log(
            (pos_sum_ba[valid_ba] + eps) / (all_sum_ba[valid_ba] + eps)
        ).mean()
    else:
        loss_ba = torch.tensor(0.0, device=z_alpha.device)

    return lambda_param * loss_ab + (1 - lambda_param) * loss_ba
