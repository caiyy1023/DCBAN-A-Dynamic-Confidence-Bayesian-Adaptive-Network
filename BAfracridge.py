"""

"""
from random import sample
import numpy as np
from numpy import interp
import warnings
import collections

from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.utils.validation import (check_X_y, check_array, check_is_fitted,
                                      _check_sample_weight)
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from sklearn.model_selection import GridSearchCV

# Module-wide constants
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.2

__all__ = ["BAfracridge", "vec_len", "BAFracRidgeRegressor",
           "BAFracRidgeRegressorCV"]

import sys
sys.path.append('/home/dell222/data/anaconda3/envs/ldm/lib/python3.8/site-packages/fracridge/')
from methods.nestedlora import NestedLoRA
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from models.builder import get_resnet_backbone  # 导入ResNet模型
from models.siam import SiamNetwork  # 导入Siam网络
from models.wide_resnet_nef import WideResNet  # 导入WideResNet模型
from models.mlp import get_mlp_eigfuncs  # 导入MLP模型构建方法

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.rand(input_dim))  # 学习到的注意力权重

    def forward(self, x):
        # 计算注意力加权和
        weighted_sum = torch.sum(x * self.attention_weights, dim=-1, keepdim=True)
        return weighted_sum

# 定义一个神经网络 SVD 函数，用于跨域检索任务
def neural_svd(X, Y, model_type='mlp', backbone_arch='resnet50', epochs=1000, output_dim=688):
    # 定义输入矩阵 X 和 Y 的维度
    input_dim = X.shape[1]
    hidden_dim = 1024  # 可以调整的隐藏层维度

    # 将 NumPy 数组转换为 PyTorch Tensor
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # 数据和模型移动到 GPU 上，以加速训练过程
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    Y = Y.to(device)

    # 使用 PCA 对 Y 进行降维，并作为训练目标
    pca = PCA(n_components=output_dim)#output_dim是降维后的维度
    Y_numpy = Y.cpu().numpy()
    target_matrix_pca = pca.fit_transform(Y_numpy)
    target_matrix_pca = torch.tensor(target_matrix_pca, dtype=torch.float32).to(device)
    #注意力机制
    attention_layer = AttentionLayer(output_dim)
    target_matrix = attention_layer(target_matrix_pca)

    # 根据不同的模型类型选择模型
    if model_type == 'mlp':
        model = get_mlp_eigfuncs(input_dim, output_dim, mlp_hidden_dims='512,512', nonlinearity='relu')
    elif model_type == 'resnet':
        backbone = get_resnet_backbone(backbone_arch)
        projector = nn.Linear(backbone.output_dim, output_dim)
        model = SiamNetwork(backbone, projector)
    elif model_type == 'wideresnet':
        model = WideResNet(depth=28, num_classes=output_dim, widen_factor=10, dropRate=0.0)
    else:
        raise ValueError("Invalid model type. Choose from ['mlp', 'resnet', 'wideresnet'].")

    model = model.to(device)
    # 初始化 SVD 解码器
    neigs = min(X.shape[0], X.shape[1])  # 保证奇异值的个数不超过矩阵的维度
    nested_lora = NestedLoRA(model, neigs)  # 替换成合适的低秩近似类

    # 定义损失函数和优化器
    optimizer = optim.Adam(nested_lora.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()  # 使用均方误差损失

    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播，计算奇异向量
        output = nested_lora(X)  # 这里的 output 的形状应与 output_dim 匹配

        # 计算损失
        loss = loss_fn(output, target_matrix)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 打印每个 epoch 的损失
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

    # 训练完成后，估计奇异值和奇异向量
    final_output = nested_lora(X)

    # 估计奇异值，长度为 output_dim
    estimated_singular_values = torch.norm(final_output, dim=0)

    # 估计右奇异向量
    estimated_right_singular_vectors = final_output / estimated_singular_values

    # 估计左奇异向量
    estimated_left_singular_vectors = torch.matmul(X.T, estimated_right_singular_vectors)

    # 归一化左奇异向量
    left_singular_vector_norms = torch.norm(estimated_left_singular_vectors, dim=1)
    estimated_left_singular_vectors = estimated_left_singular_vectors / left_singular_vector_norms.unsqueeze(1)

    # 打印特征值的形状
    print("Estimated Singular Values Shape:", estimated_singular_values.shape)

    return estimated_right_singular_vectors, estimated_singular_values, estimated_left_singular_vectors




def _do_svd(X, y):
    """
    Helper function to produce SVD outputs
    """


    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    if X.shape[0] > X.shape[1]:
        uu, ss, v_t = neural_svd(X.T @ X,y)
        selt = np.sqrt(ss)
        if y.shape[-1] >= X.shape[0]:
            ynew = (np.diag(1./selt) @ v_t @ X.T) @ y
        else:
            ynew = np.diag(1./selt) @ v_t @ (X.T @ y)

    else:
        # This rotates the targets by the unitary matrix uu.T:
        uu, selt, v_t = neural_svd(X,y)

        y = y.cpu()
        y = y.numpy()
        uu_detached = uu.detach()
        uu_cpu = uu_detached.cpu()
        uu = uu_cpu.numpy()

        selt_detached = selt.detach()
        selt_cpu = selt_detached.cpu()
        selt = selt_cpu.numpy()

        v_t_detached = v_t.detach()
        v_t_cpu = v_t_detached.cpu()
        v_t = v_t_cpu.numpy()


        ynew = uu.T @ y


    ols_coef = (ynew.T / selt).T

    return selt, v_t, ols_coef


def BAfracridge(X, y, fracs=None, tol=1e-10, jit=True):


    if fracs is None:
        fracs = np.arange(.1, 1.1, .1)

    fracs_is_vector = hasattr(fracs, "__len__")
    if fracs_is_vector:
        if np.any(np.diff(fracs) < 0):
            raise ValueError("The `frac` inputs to the `fracridge` function ",
                             f"must be sorted. You provided: {fracs}")
    else:
        fracs = [fracs]
    fracs = np.array(fracs)

    nn, pp = X.shape
    single_target = len(y.shape) == 1
    if single_target:
        y = y[:, np.newaxis]

    bb = y.shape[-1]
    ff = fracs.shape[0]

    # Calculate the rotation of the data
    selt, v_t, ols_coef = _do_svd(X, y)
    # print('selt', selt.shape)

    X_cpu = X.cpu()
    X = X_cpu.numpy()
    y_cpu = y.cpu()
    y = y_cpu.numpy()

    # Set solutions for small eigenvalues to 0 for all targets:
    isbad = selt < tol
    if np.any(isbad):
        warnings.warn("Some eigenvalues are being treated as 0")

    ols_coef[isbad, ...] = 0

    # Limits on the grid of candidate alphas used for interpolation:
    val1 = BIG_BIAS * selt[0] ** 2
    val2 = SMALL_BIAS * selt[-1] ** 2

    # Generates the grid of candidate alphas used in interpolation:
    alphagrid = np.concatenate(
        [np.array([0]),
         10 ** np.arange(np.floor(np.log10(val2)),
                         np.ceil(np.log10(val1)), BIAS_STEP)])

    # The scaling factor applied to coefficients in the rotated space is
    # lambda**2 / (lambda**2 + alpha), where lambda are the singular values
    seltsq = selt**2
    sclg = seltsq / (seltsq + alphagrid[:, None])
    sclg_sq = sclg**2

    # Prellocate the solution:
    if nn >= pp:
        first_dim = pp
    else:
        first_dim = nn

    coef = np.empty((first_dim, ff, bb))
    alphas = np.empty((ff, bb))

    # The main loop is over targets:
    for ii in range(y.shape[-1]):
        # Applies the scaling factors per alpha
        newlen = np.sqrt(sclg_sq @ ols_coef[..., ii]**2).T
        # Normalize to the length of the unregularized solution,
        # because (alphagrid[0] == 0)
        newlen = (newlen / newlen[0])
        # Perform interpolation in a log transformed space (so it behaves
        # nicely), avoiding log of 0.
        temp = interp(fracs, newlen[::-1], np.log(1 + alphagrid)[::-1])
        # Undo the log transform from the previous step
        targetalphas = np.exp(temp) - 1
        # Allocate the alphas for this target:
        alphas[:, ii] = targetalphas
        # Calculate the new scaling factor, based on the interpolated alphas:
        sc = seltsq / (seltsq + targetalphas[np.newaxis].T)
        # Use the scaling factor to calculate coefficients in the rotated
        # space:

        coef[..., ii] = (sc * ols_coef[..., ii]).T


    # After iterating over all targets, we unrotate using the unitary v
    # matrix and reshape to conform to desired output:
    print('first_dim',first_dim)
    print('ff', ff)
    print('bb', bb)
    print('pp',pp)
    print('coef',coef.shape)
    print('v_t',v_t.shape)

    coef = np.reshape(v_t @ coef.reshape((first_dim, ff * bb)),
                      (pp, ff, bb))
    print("coef.shape",coef.shape)
    if single_target:
        coef = coef.squeeze(2)
    if not fracs_is_vector:
        coef = coef.squeeze(1)
    return coef, alphas


class BAFracRidgeRegressor(BaseEstimator, MultiOutputMixin):
    
    def __init__(self, fracs, fit_intercept=True, normalize=False, copy_X=True, tol=1e-10, jit=True):
        super().__init__()
        self.fracs = fracs
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.jit = jit

    def _validate_input(self, X, y, sample_weight=None):
        """
        Helper function to validate the inputs
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight,
            check_input=True)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            outs = _rescale_data(X, y, sample_weight)
            X, y = outs[0], outs[1]

        return X, y, X_offset, y_offset, X_scale

    def fit(self, X, y, sample_weight=None):


        X, y, X_offset, y_offset, X_scale = self._validate_input(
            X, y, sample_weight=sample_weight)

        X = torch.tensor(X, dtype=torch.float32)  # 如果 X 是 numpy 数组，转换为 tensor
        y = torch.tensor(y, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        y = y.to(device)

        coef, alpha = BAfracridge(X, y, fracs=self.fracs, tol=self.tol,
                                jit=self.jit)
        self.alpha_ = alpha
        self.coef_ = coef
        self._set_intercept(X_offset, y_offset, X_scale)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        if len(self.coef_.shape) == 0:
            pred_coef = self.coef_[np.newaxis]
        else:
            pred_coef = self.coef_
        pred = np.tensordot(X, pred_coef, axes=(1))
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_
        """
        if self.fit_intercept:
            if len(self.coef_.shape) <= 1:
                self.coef_ = self.coef_ / X_scale
            elif len(self.coef_.shape) == 2:
                self.coef_ = self.coef_ / X_scale[:, np.newaxis]
            elif len(self.coef_.shape) == 3:
                self.coef_ = self.coef_ / X_scale[:, np.newaxis, np.newaxis]
            self.intercept_ = y_offset - np.tensordot(X_offset,
                                                      self.coef_, axes=(0,0))
        else:
            self.intercept_ = 0.

    def score(self, X, y, sample_weight=None):
        """
        Score the fracridge fit
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        if len(y_pred.shape) > len(y.shape):
            y = y[..., np.newaxis]
        y = np.broadcast_to(y, y_pred.shape)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {'multioutput': True}


class BAFracRidgeRegressorCV(BAFracRidgeRegressor):
   
    def __init__(self, fit_intercept=False, normalize=False,
                 copy_X=True, tol=1e-10, jit=True, cv=None, scoring=None):

        super().__init__(self, fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, tol=tol, jit=jit)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, sample_weight=None, frac_grid=None):
        """
        Parameters
        ----------
        frac_grid : sequence or float, optional
            The values of frac to consider. Default: np.arange(.1, 1.1, .1)
        """
        X, y, _, _, _ = self._validate_input(
            X, y, sample_weight=None)

        if frac_grid is None:
            frac_grid=np.arange(.1, 1.1, .1)
        parameters = {'fracs': frac_grid}
        gs = GridSearchCV(
                BAFracRidgeRegressor(
                    fit_intercept=self.fit_intercept,
                    normalize=self.normalize,
                    copy_X=self.copy_X,
                    tol=self.tol,
                    jit=self.jit),
                parameters, cv=self.cv, scoring=self.scoring)

        gs.fit(X, y, sample_weight=sample_weight)
        estimator = gs.best_estimator_
        self.best_score_ = gs.best_score_
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.best_frac_ = estimator.fracs
        self.alpha_ = estimator.alpha_
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def _more_tags(self):
        return {'multioutput': True}


def vec_len(vec, axis=0):
    return np.sqrt((vec * vec).sum(axis=axis))
