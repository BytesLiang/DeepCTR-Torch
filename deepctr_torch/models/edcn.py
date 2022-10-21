# -*- coding:utf-8 -*-
"""
Author:
    chen_kkkk, bgasdo36977@gmail.com

    zanshuxun, zanshuxun@aliyun.com
Reference:
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)

    [2] Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input, SparseFeat, VarLenSparseFeat, DenseFeat
from ..layers import CrossLayer, DNN, BridgeModule, RegulationModule


class EDCN(BaseModel):
    """Instantiates the Deep&Cross Network architecture. Including DCN-V (parameterization='vector')
    and DCN-M (parameterization='matrix').

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param cross_num: positive integet,cross layer number
    :param cross_parameterization: str, ``"vector"`` or ``"matrix"``, how to parameterize the cross network.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_cross: float. L2 regularizer strength applied to cross net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, cross_parameterization='vector',
                 bridge_type="hadamard_product",
                 dnn_hidden_units=(128, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_cross=0.00001,
                 l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task='binary', device='cpu', gpus=None):

        super(EDCN, self).__init__(linear_feature_columns=linear_feature_columns,
                                   dnn_feature_columns=dnn_feature_columns, l2_reg_embedding=l2_reg_embedding,
                                   init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        input_dim, filed_num, embedding_size = self.compute_input_dim(dnn_feature_columns, include_dense=False)
        self.cross_num = cross_num
        self.dnn_layers = nn.ModuleList([DNN(input_dim,
                                             hidden_units=[input_dim],
                                             activation=dnn_activation,
                                             use_bn=dnn_use_bn,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                             init_std=init_std, device=device)
                                         for _ in range(cross_num)])

        self.dnn_linear = nn.Linear(3 * input_dim, 1, bias=False).to(device)
        self.cross_layers = nn.ModuleList([CrossLayer(in_features=input_dim, parameterization=cross_parameterization,
                                                      device=device) for i in range(self.cross_num)])
        self.bridge_modules = nn.ModuleList([BridgeModule(input_dim, bridge_type) for _ in range(cross_num)])
        self.regulation_modules = nn.ModuleList([RegulationModule(num_fields=filed_num, embedding_dim=embedding_size,
                                                                  tau=1, use_bn=True) for _ in range(cross_num)])
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn_layers[0].named_parameters()),
            l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)
        self.add_regularization_weight(self.cross_layers[0].kernels, l2=l2_reg_cross)
        self.to(device)

    def forward(self, X):

        logit = self.linear_model(X)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        dnn_input = combined_dnn_input(sparse_embedding_list, [])
        cross_i, deep_i = self.regulation_modules[0](dnn_input)
        cross_0 = cross_i
        for i in range(self.cross_num):
            cross_i = self.cross_layers[i](cross_0, cross_i)
            deep_i = self.dnn_layers[i](deep_i)
            bridge_i = self.bridge_modules[i](cross_i, deep_i)
            if i + 1 < self.cross_num:
                cross_i, deep_i = self.regulation_modules[i + 1](bridge_i)

        stack_out = torch.cat([cross_i, deep_i, bridge_i], dim=-1)
        logit += self.dnn_linear(stack_out)
        y_pred = self.out(logit)
        return y_pred

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        filed_size = len(self.embedding_dict)
        embedding_size = sparse_feature_columns[0].embedding_dim
        return input_dim, filed_size, embedding_size
