"""
yolo核心模块
1、DarkNet:      darknet网络模型
2、yolo_loss:    yolo损失函数
3、yolo_core:    yolo模型一个用用模块，用于对网络输出做放缩处理

"""

from keras.models import Model
from keras.layers import *
import keras
from keras.regularizers import *
import keras.backend as K
import config
from keras.activations import relu


class DarkNet:
    def conv_base_block(self, inputs, filters, kernel_size, strides=(1, 1), use_bias=False):
        """
        darknet 自定义 conv层
        """
        if strides == (2, 2):
            padding = 'valid'
        else:
            padding = 'same'
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, kernel_regularizer=l2(5e-4))(inputs)
        return x

    def conv_block(self, inputs, filters, kernel_size, strides=(1, 1)):
        """
        darknet 基础组合层（conv + bn + leakyrelu）
        """
        x = self.conv_base_block(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def res_block(self, inputs, filters, block_num):
        """
        darknet 基础残差块
        """
        x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        x = self.conv_block(inputs=x, filters=filters, kernel_size=3, strides=(2, 2))
        for i in range(block_num):
            y = self.conv_block(inputs=x, filters=filters // 2, kernel_size=1)
            y = self.conv_block(inputs=y, filters=filters, kernel_size=3)
            x = Add()([x, y])
        return x

    def output_block(self, inputs, filters, output_filters):
        """
        darknet输出层
        """
        x = self.conv_block(inputs=inputs, filters=filters, kernel_size=1)
        x = self.conv_block(inputs=x, filters=filters * 2, kernel_size=3)
        x = self.conv_block(inputs=x, filters=filters, kernel_size=1)
        x = self.conv_block(inputs=x, filters=filters * 2, kernel_size=3)
        x = self.conv_block(inputs=x, filters=filters, kernel_size=1)

        y = self.conv_block(inputs=x, filters=filters * 2, kernel_size=3)
        y = self.conv_base_block(inputs=y, filters=output_filters, kernel_size=1, use_bias=True)
        return x, y

    def get_darknet(self,
                    n_class: int,
                    n_anchor: int):
        n_anchor = n_anchor // 3
        img = Input((None, None, 3))
        x = self.conv_block(inputs=img, filters=32, kernel_size=3)
        x = self.res_block(inputs=x, filters=64, block_num=1)
        x = self.res_block(inputs=x, filters=128, block_num=2)
        x = self.res_block(inputs=x, filters=256, block_num=8)
        x = self.res_block(inputs=x, filters=512, block_num=8)
        x = self.res_block(inputs=x, filters=1024, block_num=4)
        base_model = keras.models.Model(img, x)

        # o1
        x, y1 = self.output_block(inputs=x, filters=512, output_filters=n_anchor * (n_class + 5))

        x = self.conv_block(inputs=x, filters=256, kernel_size=1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, base_model.layers[152].output])
        # o2
        x, y2 = self.output_block(inputs=x, filters=256, output_filters=n_anchor * (n_class + 5))

        x = self.conv_block(inputs=x, filters=128, kernel_size=1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, base_model.layers[92].output])

        # o3
        x, y3 = self.output_block(inputs=x, filters=128, output_filters=n_anchor * (n_class + 5))

        model = keras.models.Model(img, [y1, y2, y3])
        return model

    def __call__(self,
                 n_class: int,
                 n_anchor: int,
                 *args, **kwargs):
        return self.get_darknet(n_class, n_anchor)


def yolo_core(feats, anchors, num_classes, input_shape, calc_loss=False):
    """

    :param feats:           (N, 13, 13, 3 * (5+n_class)), ...
    :param anchors:         (3, 2)
    :param num_classes:     15
    :param input_shape:     (416, 416)
    :param calc_loss:
    :return:
    """
    # 3
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    # (1, 1, 1, 3, 2)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    # (13, 13)
    grid_shape = K.shape(feats)[1:3]  # height, width
    #
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    # (13, 13, 1, 2)
    grid = K.cast(grid, K.floatx())

    # (N, 13, 13, 3 * 15)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    # 核心计算方法,
    # https://pjreddie.com/media/files/papers/YOLOv3.pdf   2.1
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # (N, 13, 13, 3, 2), (N, 13, 13, 3, 2), (N, 13, 13, 3, 1), (N, 13, 13, 3, 10)
    return box_xy, box_wh, box_confidence, box_class_probs


class CRNN:

    def __init__(self):
        self.data_format = 'channels_first'
        self.kernel_size = [3, 3, 3, 3, 3, 3, 2]
        self.paddings = [1, 1, 1, 1, 1, 1, 0]
        self.filters = [64, 128, 256, 256, 512, 512, 512]

    def base_block(self, x, i, use_BN=False):
        x = Conv2D(filters=self.filters[i],
                   kernel_size=self.kernel_size[i],
                   padding='valid' if self.paddings[i] == 0 else 'same',
                   data_format=self.data_format,
                   name='cnn.conv{0}'.format(i)
                   )(x)

        if use_BN:
            x = BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1, name='cnn.batchnorm{0}'.format(i))(x)

        x = Activation(relu, name='relu{0}'.format(i))(x)
        return x

    def __call__(self, *args, **kwargs):

        x_input = Input(shape=(1, 32, None), name='imgInput')

        x = self.base_block(x_input, 0)
        x = MaxPool2D(name='cnn.pooling{0}'.format(0), data_format=self.data_format)(x)

        x = self.base_block(x, 1)
        x = MaxPool2D(name='cnn.pooling{0}'.format(1), data_format=self.data_format)(x)

        x = self.base_block(x, 2, use_BN=True)
        x = self.base_block(x, 3)
        x = ZeroPadding2D(padding=(0, 1), data_format=self.data_format)(x)
        x = MaxPool2D(strides=(2, 1), name='cnn.pooling{0}'.format(2), data_format=self.data_format)(x)

        x = self.base_block(x, 4, use_BN=True)
        x = self.base_block(x, 5)
        x = ZeroPadding2D(padding=(0, 1), data_format=self.data_format)(x)
        x = MaxPool2D(strides=(2, 1), name='cnn.pooling{0}'.format(3), data_format=self.data_format)(x)

        x = self.base_block(x, 6, use_BN=True)

        # (N, 1, 32, None) --->>> (N, None, 32, 1)
        x = Permute((3, 2, 1))(x)

        x = Reshape((-1, 512))(x)

        x = Bidirectional(LSTM(256, return_sequences=True, recurrent_activation='sigmoid'))(x)
        x = TimeDistributed(Dense(256))(x)
        x = Bidirectional(LSTM(256, return_sequences=True, recurrent_activation='sigmoid'))(x)

        y = TimeDistributed(Dense(len(config.alphabet)+1))(x)
        return keras.models.Model(x_input, y)