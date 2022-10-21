import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import activation_layer


class LocalActivationUnit(nn.Module):
    """The LocalActivationUnit used in DIN with which the representation of
        user interests varies adaptively given different candidate items.

    Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

    Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

    Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0, dice_dim=3,
                 l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavior_len = user_behavior.size(1)

        queries = query.expand(-1, user_behavior_len, -1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior],
                                    dim=-1)  # as the source code, subtraction simulates verctors' difference
        attention_output = self.dnn(attention_input)

        attention_score = self.dense(attention_output)  # [B, T, 1]

        return attention_score


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation,
            groups, bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = F.conv2d(x, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out


class BridgeModule(nn.Module):
    def __init__(self, hidden_dim, bridge_type="hadamard_product"):
        super(BridgeModule, self).__init__()
        assert bridge_type in ["hadamard_product", "pointwise_addition", "concatenation", "attention_pooling"], \
            "bridge_type={} is not supported.".format(bridge_type)
        self.bridge_type = bridge_type
        if bridge_type == "concatenation":
            self.concat_pooling = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                                nn.ReLU())
        elif bridge_type == "attention_pooling":
            self.attention1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.Softmax(dim=-1))
            self.attention2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim, bias=False),
                                            nn.Softmax(dim=-1))

    def forward(self, X1, X2):
        out = None
        if self.bridge_type == "hadamard_product":
            out = X1 * X2
        elif self.bridge_type == "pointwise_addition":
            out = X1 + X2
        elif self.bridge_type == "concatenation":
            out = self.concat_pooling(torch.cat([X1, X2], dim=-1))
        elif self.bridge_type == "attention_pooling":
            out = self.attention1(X1) * X1 + self.attention1(X2) * X2
        return out


class RegulationModule(nn.Module):
    def __init__(self, num_fields, embedding_dim, tau=1, use_bn=False):
        super(RegulationModule, self).__init__()
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.use_bn = use_bn
        self.g1 = nn.Parameter(torch.ones(num_fields))
        self.g2 = nn.Parameter(torch.ones(num_fields))
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(num_fields * embedding_dim)
            self.bn2 = nn.BatchNorm1d(num_fields * embedding_dim)

    def forward(self, X):
        g1 = (self.g1 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).view(1, -1)
        g2 = (self.g2 / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.embedding_dim).view(1, -1)
        out1, out2 = g1 * X, g2 * X
        if self.use_bn:
            out1, out2 = self.bn1(out1), self.bn2(out2)
        return out1, out2