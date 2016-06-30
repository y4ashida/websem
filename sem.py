from __future__ import print_function
from __future__ import division
import numpy
from scipy import optimize


class Objective(object):
    def __init__(self, n, alpha, sigma, S, alpha_fixed, sigma_fixed):
        self.n = n
        self.alpha = alpha
        self.sigma = sigma
        self.S = S
        self.alpha_fixed = alpha_fixed
        self.sigma_fixed = sigma_fixed

    def __call__(self, phi):
        n_v = len(self.S)
        A, Sigma_e = self.make_matrix(phi)
        Sigma = self.Sigma(A, Sigma_e)
        e = (self.S - Sigma) * numpy.tri(n_v)
        e.shape = (n_v * n_v,)
        return e

    def make_matrix(self, x):
        n = self.n
        it = iter(x)
        A = numpy.zeros([n, n])
        Sigma_e = numpy.zeros([n, n])
        for i, j in self.alpha:
            A[i, j] = next(it)
        for i, j, val in self.alpha_fixed:
            A[i, j] = val
        for i, j in self.sigma:
            Sigma_e[i, j] = Sigma_e[j, i] = next(it)
        for i, j, val in self.sigma_fixed:
            Sigma_e[i, j] = Sigma_e[j, i] = val
        return A, Sigma_e

    def Sigma(self, A, Sigma_e):
        n_v = len(self.S)
        I = numpy.identity(self.n)
        U = numpy.zeros((n_v, self.n))
        U[:, :n_v] = numpy.identity(n_v)
        T = numpy.linalg.inv(I - A)
        return numpy.dot(
            numpy.dot(
                numpy.dot(
                    numpy.dot(U, T),
                    Sigma_e),
                T.T),
            U.T)


def calc_f_ML(Sigma, S, n):
    """
    f_ML(目的関数)を計算する

    Args:
        S:     観測値
        Sigma: 母数
        n:     観測変数の数

    Returns:
        f_ML(目的関数)
    """
    SigmaInv = numpy.linalg.inv(Sigma)
    SigmaS = numpy.dot(SigmaInv, S)

    return numpy.trace(SigmaS) - numpy.log(numpy.linalg.det(SigmaS)) - n


def calc_chi_squared(Sigma, S, n, N = 100):
    """
    χ^2を計算する

    Args:
        Sigma: 母数
        S:     観測値
        n:     観測変数の数
        N:     標本数(観測対象の数)

    Returns:
        χ^2
    """
    return (N - 1) * calc_f_ML(Sigma, S, n)


def calc_df(n, p):
    """
    df(自由度)を計算する

    Args:
        n:   観測変数の数
        p:   推定する母数の数

    Returns:
        df(自由度)
    """
    return 0.5 * n * (n + 1) - p


def calc_gfi(Sigma, S):
    n = len(Sigma)
    I = numpy.identity(n)
    SigmaInv = numpy.linalg.inv(Sigma)
    SigmaS = numpy.dot(SigmaInv, S)
    denom = numpy.trace(numpy.dot(SigmaS, SigmaS.T))
    numer = numpy.trace(numpy.dot(SigmaS - I, (SigmaS - I).T))
    return 1 - numer / denom


def calc_agfi(gfi, p, n):
    """
    AGFIを計算する

    Args:
        gfi: 適合度
        p:   推定する母数の数
        n:   観測変数の数

    Returns:
        AGFI(自由度調整済み適合度指標)
    """
    denom = 2 * calc_df(n, p)
    numer = n * (n + 1) * (1 - gfi)
    return 1 - numer / denom


def calc_cfi(Sigma, S, p, n, N = 100):
    """
    CFI(比較適合度指標)を計算する

    Args:
        Sigma: 母数
        S:     観測値
        p:     推定する母数の数
        n:     観測変数の数
        N:     標本数(観測対象の数)

    Returns:
        CFI(比較適合度指標)
    """
    df = calc_df(n, p)
    f_ML = calc_f_ML(Sigma, S, n)
    df_0 = 0.5 * n * (n - 1)
    f_0 = -numpy.log(numpy.linalg.det(numpy.diag(S) ** (-1) * S))
    numer = numpy.maximum((N - 1) * f_ML - df, 0)
    denom = numpy.maximum((N - 1) * f_0 - df_0, numer)
    return 1 - numer / denom


def calc_aic(Sigma, S, p, n, N = 100):
    """
    AIC(赤池情報量基準)を計算する

    Args:
        Sigma: 母数
        S:     観測値
        p:     推定する母数の数
        n:     観測変数の数
        N:     標本数(観測対象の数)

    Returns:
        AIC(赤池情報量基準)
    """
    chi_squared = calc_chi_squared(Sigma, S, n, N)
    df = calc_df(n, p)
    return chi_squared - df


def sem(n, alpha, sigma, S, alpha_fixed=None, sigma_fixed=None):
    if alpha_fixed is None:
        alpha_fixed = []
    if sigma_fixed is None:
        sigma_fixed = []

    num_of_params = len(alpha) + len(sigma)
    x0 = numpy.ones(num_of_params) / 10
    obj = Objective(n, alpha, sigma, S, alpha_fixed, sigma_fixed)
    if len(x0) > 0:
        sol = optimize.leastsq(obj, x0)[0]
        A, Sigma_e = obj.make_matrix(sol)
    else:
        A, Sigma_e = obj.make_matrix(x0)
    Sigma = obj.Sigma(A, Sigma_e)
    gfi = calc_gfi(Sigma, S)

    num_of_obs_vars = numpy.array(S).shape[0]
    agfi = calc_agfi(gfi, num_of_params, num_of_obs_vars)
    cfi  = calc_cfi(Sigma, S, num_of_params, num_of_obs_vars)
    aic  = calc_aic(Sigma, S, num_of_params, num_of_obs_vars)

    return A, Sigma_e, gfi, agfi, cfi, aic
