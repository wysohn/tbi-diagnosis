import keras
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
    Dropout
)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from keras import Model
from tensorflow.keras import layers
from tensorflow.keras import models
from losses import *
from generator import *


def build_unet_model(input_layer, start_filters, dropout):
    # 128 -> 64
    conv1 = Conv2D(start_filters * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_filters * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_filters * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_filters * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_filters * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_filters * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_filters * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_filters * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    # Middle
    convm = Conv2D(start_filters * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_filters * 16, (3, 3), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout)(uconv4)
    uconv4 = Conv2D(start_filters * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_filters * 8, (3, 3), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_filters * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout)(uconv3)
    uconv3 = Conv2D(start_filters * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_filters * 4, (3, 3), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(dropout)(uconv2)
    uconv2 = Conv2D(start_filters * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_filters * 2, (3, 3), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Conv2D(start_filters * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_filters * 1, (3, 3), activation="relu", padding="same")(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer


def build_unet(input_shape):
    """
    Build a UNET model

    Args:
        input_shape (tuple): the size of input images
    Returns:
        mode: a tensorflow model
    """
    input_layer = Input(input_shape)
    output_layer = build_unet_output(input_layer, 16)
    model = Model(input_layer, output_layer)
    
    return model


def down_conv_block(m, filter_mult, filters, kernel_size, name=None):
    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization()(m)
    
    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization(name=name)(m)
    
    return m


def up_conv_block(m, prev, filter_mult, filters, kernel_size, prev_2=None, prev_3=None, prev_4=None, name=None):
    m = layers.Conv2DTranspose(filter_mult * filters, kernel_size, strides=(2, 2), padding='same', activation='relu')(m)
    m = layers.BatchNormalization()(m)
    
    # concatenate layers, varies between UNet and UNet++
    if prev_4 is not None:
        m = layers.Concatenate()([m, prev, prev_2, prev_3, prev_4])
    elif prev_3 is not None:
        m = layers.Concatenate()([m, prev, prev_2, prev_3])
    elif prev_2 is not None:
        m = layers.Concatenate()([m, prev, prev_2])
    else:
        m = layers.Concatenate()([m, prev])
    
    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization(name=name)(m)
    
    return m


def build_unet(model_input, filters, kernel_size):
    # Downsampling / encoding portion
    conv1 = down_conv_block(model_input, 1, filters, kernel_size)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = down_conv_block(model_input, 2, filters, kernel_size)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = down_conv_block(model_input, 4, filters, kernel_size)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = down_conv_block(model_input, 8, filters, kernel_size)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    
    # middle of the network
    convm = down_conv_block(pool4, 16, filters, kernel_size)
    
    # upsampling / decoding portion
    uconv4 = up_conv_block(convm, conv4, 8, filters, kernel_size)
    
    uconv3 = up_conv_block(uconv4, conv3, 4, filters, kernel_size)
    
    uconv2 = up_conv_block(uconv3, conv2, 2, filters, kernel_size)
    
    uconv1 = up_conv_block(uconv2, conv1, 1, filters, kernel_size)
    
    return uconv1


def build_unet_plus_plus(model_input, filters, kernel_size, level):
    # variables names follow the UNet++ paper: [successively downsampled layers_successively upsampled layers)
    # First stage of backbone: downsampling
    conv0_0 = down_conv_block(model_input, 1, filters, kernel_size, name='conv0_0')
    pool0_0 = layers.MaxPooling2D((2, 2))(conv0_0)
    conv1_0 = down_conv_block(pool0_0, 2, filters, kernel_size, name='conv1_0')

    if level > 1:
        # Second stage
        pool1_0 = layers.MaxPooling2D((2, 2))(conv1_0)
        conv2_0 = down_conv_block(pool1_0, 4, filters, kernel_size, name='conv2_0')

        if level > 2:
            # Third stage
            pool2_0 = layers.MaxPooling2D((2, 2))(conv2_0)
            conv3_0 = down_conv_block(pool2_0, 8, filters, kernel_size, name='conv3_0')

            if level > 3:
                # Fourth stage
                pool3_0 = layers.MaxPooling2D((2, 2))(conv3_0)
                conv4_0 = down_conv_block(pool3_0, 16, filters, kernel_size, name='conv4_0')

    # First stage of upsampling and skip connections
    conv0_1 = up_conv_block(conv1_0, conv0_0, 1, filters, kernel_size, name='conv0_1')
    out = conv0_1

    if level > 1:
        # Second stage
        conv1_1 = up_conv_block(conv2_0, conv1_0, 2, filters, kernel_size, name='conv1_1')
        conv0_2 = up_conv_block(conv1_1, conv0_1, 1, filters, kernel_size, prev_2=conv0_0, name='conv0_2')
        out = conv0_2

        if level > 2:
            # Third stage
            conv2_1 = up_conv_block(conv3_0, conv2_0, 4, filters, kernel_size, name='conv2_1')
            conv1_2 = up_conv_block(conv2_1, conv1_1, 2, filters, kernel_size, prev_2=conv1_0, name='conv1_2')

            conv0_3 = up_conv_block(conv1_2, conv0_2, 1, filters, kernel_size, prev_2=conv0_1, prev_3=conv0_0,
                                    name='conv0_3')
            out = conv0_3

            if level > 3:
                # Fourth stage
                conv3_1 = up_conv_block(conv4_0, conv3_0, 8, filters, kernel_size, name='conv3_1')
                conv2_2 = up_conv_block(conv3_1, conv2_1, 4, filters, kernel_size, prev_2=conv2_0, name='conv2_2')
                conv1_3 = up_conv_block(conv2_2, conv1_2, 2, filters, kernel_size, prev_2=conv1_1, prev_3=conv1_0,
                                        name='conv1_3')
                conv0_4 = up_conv_block(conv1_3, conv0_3, 1, filters, kernel_size, prev_2=conv0_2, prev_3=conv0_1,
                                        prev_4=conv0_0, name='conv0_4')
                out = conv0_4

    return out


def create_segmentation_model(input_height, input_width, architecture='unet_plus_plus', level = 3):
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
    model_input = layers.Input((input_height, input_width, 1))
    
    assert level in range(1, 5), f'UNet++ depth {level} not allowed. level must be in range: 1, 2, 3, 4.'
    
    if architecture == 'unet_plus_plus':
        model_output = build_unet_plus_plus(model_input, 32, (3, 3), level)
    elif architecture == 'unet':
        model_output = build_unet(model_input, 32, (3, 3))
    else:
        raise AttributeError(f'Network architecture {architecture} does not exist.')
    
    # the output layer
    output_layer = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='output_conv')(model_output)
    
    model = models.Model(inputs=model_input, outputs=output_layer)
    model.summary()
    
    return model