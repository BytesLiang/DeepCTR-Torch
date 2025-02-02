# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, KBinsDiscretizer

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

embedding_dim = 4


if __name__ == "__main__":
    # data = pd.read_csv('./criteo_sample.txt')
    data = pd.read_csv('./criteo_sampled_data.csv')
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1)
                                                              for feat in dense_features]
    # for feat in dense_features:
    # kbd = KBinsDiscretizer(n_bins=16, strategy="quantile", encode="ordinal")
    # data[dense_features] = kbd.fit_transform(data[dense_features])
    #
    # fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
    #                           for feat in sparse_features] + [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
    #                                                           for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # model = DCNAD(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #               task='binary', dnn_dropout=0, dense_embedding="auto_dis",
    #               l2_reg_embedding=1e-5, device=device)

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #               task='binary', dnn_dropout=0,
    #               l2_reg_embedding=1e-5, device=device)

    model = DeepFMAd(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary', dnn_dropout=0,
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    history = model.fit(train_model_input, train[target].values, batch_size=1024, epochs=1, verbose=1,
                        validation_split=0.2)

    pred_ans = model.predict(test_model_input, 256)
    # torch.save(model, r'D:\liangxinzhu1\model.pt')
    script = torch.jit.script(model)
    torch.jit.save(script, r'D:\liangxinzhu1\model2.pt')
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
