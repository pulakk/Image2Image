from keras.layers import Conv2D, Conv2DTranspose, Dense, Input, Flatten, Dropout
from keras.layers import Reshape, Activation, Concatenate, BatchNormalization

from keras.models import Model
from keras.optimizers import SGD

from functools import reduce

STRIDES = (2,2)
KERNEL = (7,7)

# LAYERS
def ConvLayer(filters, kernel = KERNEL, strides = STRIDES, activation = 'relu'):
    return Conv2D ( filters, kernel_size = kernel, padding='same', strides = strides, activation = activation )

def DeConvLayer(filters, kernel = KERNEL, strides = STRIDES, activation = 'relu'):
    return Conv2DTranspose(filters, kernel_size = kernel, padding='same', strides = strides, activation = activation )

def ConvLayers(in_image, filters, drop_rate = 0.1):
    flow = in_image
    for f in filters:
        flow = ConvLayer ( f ) ( flow )
        flow = Dropout ( drop_rate ) ( flow)
        flow = BatchNormalization () ( flow )
    return flow

def DeConvLayers(in_image, filters, drop_rate = 0.1):
    flow = in_image
    for i in range(len(filters)):
        f = filters[i]
        if i == len(filters) - 1:
            flow = DeConvLayer ( f, activation = 'sigmoid' ) ( flow ) 
        else:
            flow = DeConvLayer( f ) ( flow )
            flow = Dropout ( drop_rate ) ( flow )
            flow = BatchNormalization () ( flow )
    return flow


# Image to Image - Encoder Decoder model
def Im2Im(im_shape):
    filters = [32, 64]
    channels = [im_shape[-1]] + filters

    # generator layers
    G_in = Input ( shape = im_shape ) # input image

    gen_flow = ConvLayers ( G_in, channels [ 1: ] ) # convolution

    conv_shape = list ( map ( lambda x : int ( x ), gen_flow.shape [ 1: ] ) )
    units = int ( reduce ( lambda x, y: x * y, conv_shape ) )

    gen_flow = Flatten () ( gen_flow )
    gen_flow = Dense ( units, activation = 'relu' ) ( gen_flow ) # dense
    gen_flow = BatchNormalization () ( gen_flow )
    gen_flow = Reshape ( conv_shape ) ( gen_flow )

    gen_flow = DeConvLayers ( gen_flow, channels [ -1 :: -1 ] [ 1: ] ) #  transpose convolution
    G_out = gen_flow
    
    model = Model(G_in, G_out)
    
    sgd = SGD(lr = 10,momentum=0.1)
    model.compile(optimizer = sgd,loss = 'mse')

    return model
