from keras.layers import Input, Flatten, AveragePooling2D
from keras.models import Model
from qkeras import QDense, QActivation, QConv2D, QSeparableConv2D, quantized_bits
import tensorflow as tf
from SoftQuantizeLayer import SoftQuantizeLayer
from PairLatticeRouterLayer import PairLatticeRouterLayer

def _var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    
    # First QDense layer
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    
    # Second QDense layer
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)

    # Last QDense layer (output)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def _conv_network(var, n_filters=5, kernel_size=3):
    var = QSeparableConv2D(
        n_filters,kernel_size,
        depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,1,
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    return var
     
def QConv2D_Max(shape, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = _conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = _var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model

def QConv2D_Full(shape, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = _conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack)
    return model

def QConv2D_Slim(shape, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = _conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model

def QConv2D_Max_SoftQuantizer(shape, n_filters, pool_size, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)
    stack = _conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = _var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model
    
def QConv2D_Full_SoftQuantizer(shape, n_filters, pool_size, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)
    stack = _conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack)
    return model

def QConv2D_Slim_SoftQuantizer(shape, n_filters, pool_size, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape)
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)
    stack = _conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model


def QConv2D_Max_PairLattice(shape, n_filters, pool_size, initial_thresholds,
                            threshold_offset, initial_levels=None,
                            trainable_thresholds=True):
    """O25: the deployable quantized backbone behind the O24 front end.

    Same PairLatticeRouterLayer + SoftQuantizeLayer, under the same layer names,
    that every ViT campaign used. That is the whole point: the discovery driver's
    router/quantizer plumbing, the v2 NLL, the by-term bin balancing and all 60
    zero-bias MDMM constraints then run through ONE code path for both
    architectures, so a QConv2D-vs-ViT difference is a statement about the
    network and not about two subtly different experiments. Only the backbone
    differs -- 1,869 quantized parameters here against the ViT's ~711K float.

    Pin the readout with SMARTPIX_FIX_SLICES + SMARTPIX_PIN_MODE=onehot and
    SMARTPIX_FREEZE_THR=1; the router's per-step pair sampling is then
    degenerate by construction, which is what "fixed readout" means. The
    sampling that a one-hot pin destroyed for the ViT (runs/_o22_onehot_overfit_on)
    was load-bearing regularization for 711K parameters; at 1,869 it is not.
    """
    x_in = Input(shape, name='raw_input')
    x = PairLatticeRouterLayer(num_slots=2, smooth_logits=True,
                               name='simple_router_output')(x_in)
    x = SoftQuantizeLayer(
        n_bits=2,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,
        trainable_k=True,
        name='soft_quantizer_output',
    )(x)
    stack = _conv_network(x, n_filters)
    stack = AveragePooling2D(pool_size=(pool_size, pool_size), strides=None,
                             padding="valid", data_format=None)(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = _var_network(stack, hidden=16, output=14)
    return Model(inputs=x_in, outputs=stack, name='smrtpxl_qconv2d_pairlattice')
