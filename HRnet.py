import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, UpSampling2D
from tensorflow.keras.models import Model

# Hyperparameters
BN_MOMENTUM = 0.1
BN_EPSILON = 1e-5
INITIALIZER = 'he_normal'
num_classes = 19  # Set the number of classes as per your dataset

# Convolution Block
def conv_block(x, filters, kernel_size, strides=1, activation='relu', name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same',
               kernel_initializer=INITIALIZER, use_bias=False, name=name+'_conv')(x)
    x = BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MOMENTUM, name=name+'_bn')(x)
    if activation:
        x = Activation(activation, name=name+'_act')(x)
    return x

# Basic Residual Block
def residual_block(x, filters, strides=1, downsample=False, name=None):
    shortcut = x
    x = conv_block(x, filters, 3, strides=strides, activation='relu', name=name+'_res1')
    x = conv_block(x, filters, 3, strides=1, activation=None, name=name+'_res2')

    # Downsample shortcut if required
    if downsample:
        shortcut = conv_block(shortcut, filters, 1, strides=strides, activation=None, name=name+'_downsample')

    x = Add(name=name+'_add')([x, shortcut])
    x = Activation('relu', name=name+'_out')(x)
    return x

# HRNet Stage
def hrnet_stage(x, num_blocks, filters, stage_name):
    for i in range(num_blocks):
        x = residual_block(x, filters, downsample=(i==0), name=f'{stage_name}_block{i+1}')
    return x

# HRNet Model
def hrnet_model(input_shape=(128, 128, 3), num_classes=19):
    inputs = Input(shape=input_shape)

    # Initial layers
    x = conv_block(inputs, 64, 3, strides=2, activation='relu', name='init_conv1')
    x = conv_block(x, 64, 3, strides=2, activation='relu', name='init_conv2')

    # Stages
    x = hrnet_stage(x, num_blocks=4, filters=64, stage_name='stage1')
    
    # Transition layers and further stages can be added here as needed

    # Final upsample and classification layer
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

# Initialize and compile the model
model = hrnet_model(input_shape=(128, 128, 3), num_classes=num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

import tensorflow as tf

def IoU_per_class(y_true, y_pred, num_classes, smooth=1e-6):
    iou_list = []
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.squeeze(y_true, axis=-1)  # Remove the channel dimension from y_true

    for i in range(num_classes):
        true_labels = tf.equal(y_true, i)
        predicted_labels = tf.equal(y_pred, i)
        intersection = tf.reduce_sum(tf.cast(true_labels & predicted_labels, tf.float32))
        union = tf.reduce_sum(tf.cast(true_labels, tf.float32)) + tf.reduce_sum(tf.cast(predicted_labels, tf.float32)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou)
    return iou_list


def mIoU(y_true, y_pred, num_classes):
    iou_list = IoU_per_class(y_true, y_pred, num_classes)
    m_iou = tf.reduce_mean(iou_list)
    return m_iou

# Named metric functions for use in model.compile
def compute_mIoU(y_true, y_pred):
    return mIoU(y_true, y_pred, num_classes)

def compute_IoU_per_class(y_true, y_pred):
    return IoU_per_class(y_true, y_pred, num_classes)


