import argparse
import os
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from BAfracridge import BAFracRidgeRegressor

def main():
    parser = argparse.ArgumentParser(description="Train BAFracRidge model on fMRI and image features.")
    parser.add_argument('--fmri_train', type=str, required=True, help='Path to training fMRI features (npy file)')
    parser.add_argument('--fmri_test', type=str, required=True, help='Path to test fMRI features (npy file)')
    parser.add_argument('--image_train', type=str, required=True, help='Path to training image features (npy file)')
    parser.add_argument('--image_test', type=str, required=True, help='Path to test image features (npy file)')
    parser.add_argument('--output', type=str, default='fmri_pred.npy', help='Path to save predicted fMRI features')
    args = parser.parse_args()

    # 加载数据
    X_train = np.load(args.image_train)  # 图像特征
    X_test = np.load(args.image_test)
    Y_train = np.load(args.fmri_train)   # fMRI特征
    Y_test = np.load(args.fmri_test)

    # 定义 gamma 值范围
    gamma_values = np.linspace(0, 1, 21)

    # 构建 BAFracRidge 模型
    ridge = BAFracRidgeRegressor(fracs=gamma_values)

    # 构建预处理 + 模型的 pipeline
    preprocess_pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=True))
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )

    # 训练模型
    pipeline.fit(X_train, Y_train)

    # 预测
    Y_pred = pipeline.predict(X_test)

    # 保存预测结果
    np.save(args.output, Y_pred)
    print(f"预测的 fMRI 特征已保存到: {args.output}")

if __name__ == '__main__':
    main()
