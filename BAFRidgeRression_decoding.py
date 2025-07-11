import argparse, os
import numpy as np
import torch.cuda
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from BAfracridge import BAFracRidgeRegressor
import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--roi",
        required=True,
        type=str,
        nargs="*",
        help="use roi name",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi

    # target = "c"
    # roi = ["ventral"]


    backend = set_backend("numpy", on_error="warn")
    subject = opt.subject
    # subject = "subj01"

    # 定义γ值范围
    gamma_values = np.linspace(0, 1, 21)  # 从0到1，间隔为0.05


    ridge = BAFracRidgeRegressor(fracs=gamma_values)


    preprocess_pipeline = make_pipeline(

        StandardScaler(with_mean=True, with_std=True),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )

    import cupy as cp

    # 查看当前正在使用的设备
    device = cp.cuda.Device()
    print("Current device:", device)

    mridir = f'../../mrifeatsession1/{subject}/'
    featdir = '../../nsdfeatsession1/subjfeat/'
    savedir = f'../..//decodedFRR_neuralsvd/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    X = []
    X_te = []
    for croi in roi:
        if 'conv' in target:  # We use averaged features for GAN due to large number of dimension of features
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy').astype("float32")
        else:
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        X.append(cX)
        X_te.append(cX_te)
    X = np.hstack(X)
    X_te = np.hstack(X_te)

    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32")

    Y = Y.reshape([X.shape[0], -1])
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32").reshape([X_te.shape[0], -1])

    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')

    def fit_in_batches(model, X, y, batch_size=1):

        n_samples = X.shape[0]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            print(f"Fitting batch from {start} to {end}")
            model.fit(X[start:end], y[start:end])

    # print("X:",X)
    # print("Y:",Y)

    # 调用分批处理的拟合函数
    # fit_in_batches(pipeline, X, Y)

    pipeline.fit(X, Y)
    scores = pipeline.predict(X_te)

    rs = correlation_score(Y_te.T, scores.T[:,0,:])
    print(f'Prediction accuracy is: {np.mean(rs):3.3}')

    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy', scores)


if __name__ == "__main__":
    main()