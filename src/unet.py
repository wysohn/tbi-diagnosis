import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    Lambda,
    Concatenate,
    Input,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import models
from losses import *
from generator import *
import config


def down_conv_block(m, filter_mult, filters, kernel_size, name=None):
    m = Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    #m = layers.BatchNormalization()(m)
    
    m = Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    #m = layers.BatchNormalization(name=name)(m)
    #m = Dropout(0.5)(m)
    
    return m


def up_conv_block(m, 
                  prev, 
                  filter_mult, 
                  filters, 
                  kernel_size,
                  dropout_rate,
                  prev_2=None, 
                  prev_3=None, 
                  prev_4=None, 
                  name=None):
    m = Conv2DTranspose(filter_mult * filters, kernel_size, strides=(2, 2), padding='same', activation='relu')(m)
    #m = layers.BatchNormalization()(m)

    # Concatenate layers; varies between UNet and UNet++
    if prev_4 is not None:
        m = Concatenate()([m, prev, prev_2, prev_3, prev_4])
    elif prev_3 is not None:
        m = Concatenate()([m, prev, prev_2, prev_3])
    elif prev_2 is not None:
        m = Concatenate()([m, prev, prev_2])
    else:
        m = Concatenate()([m, prev])
        m = Dropout(dropout_rate)(m)

    m = Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)

    return m


def build_unet(model_input, filters, kernel_size, dropout_rate):
    # Downsampling / encoding portion
    conv0 = down_conv_block(model_input, 1, filters, kernel_size)
    pool0 = MaxPooling2D((2, 2))(conv0)
    pool0 = Dropout(dropout_rate)(pool0)

    conv1 = down_conv_block(pool0, 2, filters, kernel_size)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = down_conv_block(pool1, 4, filters, kernel_size)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = down_conv_block(pool2, 8, filters, kernel_size)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)

    # Middle of network
    conv4 = down_conv_block(pool3, 16, filters, kernel_size)

    # Upsampling / decoding portion
    uconv3 = up_conv_block(conv4, conv3, 8, filters, kernel_size, dropout_rate)

    uconv2 = up_conv_block(uconv3, conv2, 4, filters, kernel_size, dropout_rate)

    uconv1 = up_conv_block(uconv2, conv1, 2, filters, kernel_size, dropout_rate)

    uconv0 = up_conv_block(uconv1, conv0, 1, filters, kernel_size, dropout_rate)

    return uconv0


def build_attention_unet(model_input, filters, kernel_size, dropout_rate):
    # Downsampling / encoding portion
    conv0 = down_conv_block(model_input, 1, filters, kernel_size)
    pool0 = MaxPooling2D((2, 2))(conv0)
    pool0 = Dropout(dropout_rate)(pool0)

    conv1 = down_conv_block(pool0, 2, filters, kernel_size)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = down_conv_block(pool1, 4, filters, kernel_size)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = down_conv_block(pool2, 8, filters, kernel_size)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)

    # Middle of network
    conv4 = down_conv_block(pool3, 16, filters, kernel_size)

    # Upsampling / decoding portion
    attn3 = AttnGateBlock(conv3, conv4, filters * 16)
    uconv3 = up_conv_block(conv4, attn3, 8, filters, kernel_size, dropout_rate)

    attn2 = AttnGateBlock(conv2, uconv3, filters * 8)
    uconv2 = up_conv_block(uconv3, attn2, 4, filters, kernel_size, dropout_rate)

    attn1 = AttnGateBlock(conv1, uconv2, filters * 4)
    uconv1 = up_conv_block(uconv2, attn1, 2, filters, kernel_size, dropout_rate)

    attn0 = AttnGateBlock(conv0, uconv1, filters * 2)
    uconv0 = up_conv_block(uconv1, attn0, 1, filters, kernel_size, dropout_rate)

    return uconv0


def build_unet_plus_plus(model_input, filters, kernel_size, dropout_rate, level):
    # variables names follow the UNet++ paper: [successively downsampled layers_successively upsampled layers)
    # First stage of backbone: downsampling
    conv0_0 = down_conv_block(model_input, 1, filters, kernel_size, name='conv0_0')
    pool0_0 = MaxPooling2D((2, 2))(conv0_0)
    pool0_0 = Dropout(dropout_rate)(pool0_0)
    conv1_0 = down_conv_block(pool0_0, 2, filters, kernel_size, name='conv1_0')

    if level > 1:
        # Second stage
        pool1_0 = MaxPooling2D((2, 2))(conv1_0)
        pool1_0 = Dropout(dropout_rate)(pool1_0)
        conv2_0 = down_conv_block(pool1_0, 4, filters, kernel_size, name='conv2_0')

        if level > 2:
            # Third stage
            pool2_0 = MaxPooling2D((2, 2))(conv2_0)
            pool2_0 = Dropout(dropout_rate)(pool2_0)
            conv3_0 = down_conv_block(pool2_0, 8, filters, kernel_size, name='conv3_0')

            if level > 3:
                # Fourth stage
                pool3_0 = MaxPooling2D((2, 2))(conv3_0)
                pool3_0 = Dropout(dropout_rate)(pool3_0)
                conv4_0 = down_conv_block(pool3_0, 16, filters, kernel_size, name='conv4_0')

    # First stage of upsampling and skip connections
    conv0_1 = up_conv_block(conv1_0, conv0_0, 1, filters, kernel_size, dropout_rate, name='conv0_1')
    out = conv0_1

    if level > 1:
        # Second stage
        conv1_1 = up_conv_block(conv2_0, 
                                conv1_0, 
                                2, 
                                filters, 
                                kernel_size, 
                                dropout_rate, name='conv1_1')
        conv0_2 = up_conv_block(conv1_1, 
                                conv0_1, 
                                1, 
                                filters, 
                                kernel_size, 
                                dropout_rate, 
                                prev_2=conv0_0, 
                                name='conv0_2')
        out = conv0_2

        if level > 2:
            # Third stage
            conv2_1 = up_conv_block(conv3_0, 
                                    conv2_0, 
                                    4, 
                                    filters, 
                                    kernel_size, 
                                    dropout_rate, 
                                    name='conv2_1')
            conv1_2 = up_conv_block(conv2_1, 
                                    conv1_1, 
                                    2, 
                                    filters, 
                                    kernel_size, 
                                    dropout_rate, 
                                    prev_2=conv1_0, 
                                    name='conv1_2')

            conv0_3 = up_conv_block(conv1_2, 
                                    conv0_2, 
                                    1, 
                                    filters, 
                                    kernel_size, 
                                    dropout_rate, 
                                    prev_2=conv0_1, 
                                    prev_3=conv0_0,
                                    name='conv0_3')
            out = conv0_3

            if level > 3:
                # Fourth stage
                conv3_1 = up_conv_block(conv4_0, 
                                        conv3_0, 
                                        8, 
                                        filters, 
                                        kernel_size, 
                                        dropout_rate, 
                                        name='conv3_1')
                conv2_2 = up_conv_block(conv3_1, 
                                        conv2_1, 
                                        4, 
                                        filters, 
                                        kernel_size, 
                                        dropout_rate, 
                                        prev_2=conv2_0, 
                                        name='conv2_2')
                conv1_3 = up_conv_block(conv2_2, 
                                        conv1_2, 
                                        2, 
                                        filters, 
                                        kernel_size, 
                                        dropout_rate, 
                                        prev_2=conv1_1, 
                                        prev_3=conv1_0,
                                        name='conv1_3')
                conv0_4 = up_conv_block(conv1_3, 
                                        conv0_3, 
                                        1, 
                                        filters, 
                                        kernel_size, 
                                        dropout_rate, 
                                        prev_2=conv0_2, 
                                        prev_3=conv0_1,
                                        prev_4=conv0_0, 
                                        name='conv0_4')
                out = conv0_4

    return out


def expand_as(tensor, rep):
    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (256,80,N), lambda will return a tensor of shape (256,80,N*rep), 
    # if specified axis=2
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                       arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGateBlock(x, g, inter_shape):
    # attention gate as described in 
    #   Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, 
    #   Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard 
    #   Kainz, Ben Glocker, and Daniel Rueckert. Attention U-Net: Learning Where to 
    #   Look for the Pancreas. arXiv, page 10, 2018
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)
    
    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape, 
                   kernel_size=1, 
                   strides=1, 
                   padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape, 
                     kernel_size=3, 
                     strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]), 
                     padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = tf.math.add(phi_g, theta_x)
    add_xg = Activation('relu')(add_xg)

    # 1x1 convolution
    psi = Conv2D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], 
                                             shape_x[2] // shape_sigmoid[2])
                                      )(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expand_as(upsample_sigmoid_xg, shape_x[3])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = tf.math.multiply(upsample_sigmoid_xg, x)

    # Final 1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv2D(filters=shape_x[3], 
                    kernel_size=1, strides=1, 
                    padding='same')(attn_coefficients)
    
    output = BatchNormalization()(output)
    
    return output


def build_cascade_unet_conv(model_input, ROI_input, filters, kernel_size, dropout_rate):
    # Cascade U-Net inspired by Sagi Eppel, Haoping Xu, and Alan Aspuru-Guzik. 
    #   Computer vision for liquid samples in hospitals and medical labs using hierarchical 
    #   image segmentation and relations prediction

    # extract filters from ROI and input
    ROIconv = Conv2D(filters, (3, 3), padding="same")(ROI_input)
    
    # Downsampling / encoding portion
    conv0 = down_conv_block(model_input, 1, filters, kernel_size)
    conv0 = tf.math.multiply(conv0, ROIconv)
    pool0 = MaxPooling2D((2, 2))(conv0)
    pool0 = Dropout(dropout_rate)(pool0)

    conv1 = down_conv_block(pool0, 2, filters, kernel_size)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = down_conv_block(pool1, 4, filters, kernel_size)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = down_conv_block(pool2, 8, filters, kernel_size)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)

    # Middle of network
    conv4 = down_conv_block(pool3, 16, filters, kernel_size)

    # Upsampling / decoding portion
    uconv3 = up_conv_block(conv4, conv3, 8, filters, kernel_size, dropout_rate)

    uconv2 = up_conv_block(uconv3, conv2, 4, filters, kernel_size, dropout_rate)

    uconv1 = up_conv_block(uconv2, conv1, 2, filters, kernel_size, dropout_rate)

    uconv0 = up_conv_block(uconv1, conv0, 1, filters, kernel_size, dropout_rate)

    return uconv0


def build_cascade_unet_concat(model_input, ROI_input, filters, kernel_size, dropout_rate):
    # Cascade U-Net inspired by Yizhe Zhang, Michael T. C. Ying, Lin Yang, Anil T. Ahuja, and Danny Z. Chen. 
    #   Coarse-to-fine stacked fully convolutional nets for lymph node segmentation in ultrasound images

    # merge the ROI and input by concatenation
    merge_layer = Concatenate()([model_input, ROI_input])

    # Downsampling / encoding portion
    conv0 = down_conv_block(merge_layer, 1, filters, kernel_size)
    pool0 = MaxPooling2D((2, 2))(conv0)
    pool0 = Dropout(dropout_rate)(pool0)

    conv1 = down_conv_block(pool0, 2, filters, kernel_size)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    conv2 = down_conv_block(pool1, 4, filters, kernel_size)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)

    conv3 = down_conv_block(pool2, 8, filters, kernel_size)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)

    # Middle of network
    conv4 = down_conv_block(pool3, 16, filters, kernel_size)

    # Upsampling / decoding portion
    uconv3 = up_conv_block(conv4, conv3, 8, filters, kernel_size, dropout_rate)

    uconv2 = up_conv_block(uconv3, conv2, 4, filters, kernel_size, dropout_rate)

    uconv1 = up_conv_block(uconv2, conv1, 2, filters, kernel_size, dropout_rate)

    uconv0 = up_conv_block(uconv1, conv0, 1, filters, kernel_size, dropout_rate)

    return uconv0


def create_segmentation_model(input_height, 
                              input_width, 
                              filters, 
                              architecture, 
                              level, 
                              dropout_rate):
    """
    Create a segmentation model
    
    Args:
        input_height (int): the height of input to the segmentation model in pixels
        input_width (int): the width of input to the segmentation model in pixels
        architecture (string): 'unet' if 'unet_plus_plus'
        level (int): unet depth; the maximal number of down-convolution and up-convolution blocks
    
    Return:
        model: A segmentation model
    """
    model_input = Input((input_height, input_width, 1))
    ROI_input = Input((input_height, input_width, 1))
    
    assert level in range(1, 5), f'UNet++ depth {level} not allowed. level must be in range: 1, 2, 3, 4.'
    
    if architecture == 'unet_plus_plus':
        model_output = build_unet_plus_plus(model_input, filters, (3, 3), dropout_rate, level)
    elif architecture == 'unet':
        model_output = build_unet(model_input, filters, (3, 3), dropout_rate)
    elif architecture == 'attention_unet':
        model_output = build_attention_unet(model_input, filters, (3, 3), dropout_rate)
    elif architecture == 'cascade_unet_concat':
        model_output = build_cascade_unet_concat(model_input, ROI_input, filters, (3, 3), dropout_rate)
    elif architecture == 'cascade_unet_conv':
        model_output = build_cascade_unet_conv(model_input, ROI_input, filters, (3, 3), dropout_rate)
    else:
        raise AttributeError(f'Network architecture {architecture} does not exist.')
    
    # the output layer
    output_layer = Conv2D(
                        1,
                        (1, 1), 
                        padding='same', 
                        activation='sigmoid', 
                        name='output_conv')(model_output
                    )
    
    if architecture == 'cascade_unet_conv' or architecture == 'cascade_unet_concat':
        model = models.Model(inputs=[model_input, ROI_input], outputs=output_layer)
    else:
        model = models.Model(inputs=model_input, outputs=output_layer)
    #model.summary()
    
    return model
    