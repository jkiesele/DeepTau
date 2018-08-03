
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}


from keras import backend as K
import tensorflow as tf

def relative_rms(y_true, y_pred):
    calculate = tf.square((y_true-y_pred)/( tf.abs(y_true)+K.epsilon()))
    
    mask = tf.where(tf.abs(y_true)>0, 
                         tf.zeros_like(y_true)+1, 
                         tf.zeros_like(y_true))
    
    non_zero=tf.count_nonzero(mask,dtype='float32')
    calculate *= mask
    
    calculate = tf.reshape(calculate, [-1])
    calculate = K.sum(calculate,axis=-1)
    non_zero = tf.reshape(non_zero, [-1])
    ret = tf.sqrt(tf.abs(calculate/(non_zero+K.epsilon()))+K.epsilon())*100
    ret = tf.where(tf.is_inf(ret), tf.zeros_like(ret), ret)
    
    return ret

global_loss_list['relative_rms']=relative_rms