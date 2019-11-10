from keras.layers import Conv2D, Conv2DTranspose, Dense, Input, Flatten, Dropout
from keras.layers import Reshape, Activation, Concatenate, BatchNormalization

from keras.models import Model
from keras.optimizers import SGD

from functools import reduce

STRIDES = (1,1)
KERNEL = (7,7)

# LAYERS
def ConvLayer( filters, kernel = KERNEL, strides = STRIDES, activation = 'relu' ):
    return Conv2D ( filters, kernel_size = kernel, padding='same', strides = strides, activation = activation )

def ConvLayers(in_image, filters, drop_rate = 0.1):
    flow = in_image
    for f in filters:
        flow = ConvLayer ( f ) ( flow )
        flow = Dropout ( drop_rate ) ( flow)
        flow = BatchNormalization () ( flow )
    return flow

# Image to Image - Encoder Decoder model
def Im2Im( in_shape, out_shape ):
    filters = [32, out_shape[-1]]
    channels = [in_shape[-1]] + filters

    # generator layers
    G_in = Input ( shape = in_shape ) # input image

    gen_flow = ConvLayers ( G_in, channels [ 1: ] ) # convolution

    # conv_shape = list ( map ( lambda x : int ( x ), gen_flow.shape [ 1: ] ) )
    units = int ( reduce ( lambda x, y: x * y, out_shape ) )

    gen_flow = Flatten () ( gen_flow )
    gen_flow = Dense ( units ) ( gen_flow ) # dense
    gen_flow = BatchNormalization () ( gen_flow )
    gen_flow = Reshape ( out_shape ) ( gen_flow )

    G_out = gen_flow
    
    model = Model(G_in, G_out)
    
    sgd = SGD(lr = 0.5,momentum=0.1)
    model.compile(optimizer = sgd,loss = 'mse')
    
    model.summary()

    return model
