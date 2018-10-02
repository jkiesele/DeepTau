import tensorflow

# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}


from keras.engine.topology import Layer
import tensorflow as tf
import keras.backend as K
    
    
class SortByFirstFeatureAndPool1D(Layer):
    def __init__(self, noutputs , **kwargs):
        super(SortByFirstFeatureAndPool1D, self).__init__(**kwargs)
        self.noutputs=noutputs
    
    def get_config(self):
        config = {'noutputs': self.noutputs}
        base_config = super(SortByFirstFeatureAndPool1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    
    def compute_output_shape(self, input_shape):
        
        outshape=list(input_shape)
        outshape[1]=self.noutputs
        print('compute', tuple(outshape))
        return tuple(outshape)
    
    def call(self, inputs):
        #top_k
        # B X PF X F
        n_batch = tf.shape(inputs)[0]
        n_in = tf.shape(inputs)[1]
        
        _, I = tf.nn.top_k(tf.reduce_max(inputs, axis=2), self.noutputs)
        I = tf.expand_dims(I, axis=2)
        batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
        batch_range = tf.tile(batch_range, [1, self.noutputs, 1])
        _indexing_tensor = tf.concat([batch_range, I], axis=2)
       
        out = tf.gather_nd(inputs, _indexing_tensor)
        print('get',out.shape)
        return out
    
        #batchshape = tf.shape(inputs)[0].shape
        #print(batchshape)
        #if len(batchshape)>0:
        #    n_batch=int(batchshape[0])
        #params = inputs
        #query = tf.nn.top_k(params[:, :, 0], k=self.noutputs, sorted=True).indices
        #n = tf.shape(params)[0]
        #ii = tf.tile(tf.range(n)[:, tf.newaxis, tf.newaxis], (1, self.noutputs, 1))
        #idx = tf.concat([ii, query], axis=-1)
        #result = tf.gather_nd(params, idx)
        #print('get',result.shape)
        #return result
        #
        batch_range = tf.expand_dims(tf.expand_dims(tf.range(n_batch), axis=1), axis=1)
        batch_range = tf.tile(batch_range, [1,n_in, 1])
        _indexing_tensor = tf.concat([batch_range, I], axis=2)
        out = tf.gather_nd(inputs, _indexing_tensor)
        
        #somehow keras does not allow to use only the first k, so needs to be cut
        #out = out[:,:self.noutputs]
        print('get',out.shape)
        return out
    


class Clip(Layer):
    def __init__(self, min, max , **kwargs):
        super(Clip, self).__init__(**kwargs)
        self.min=min
        self.max=max
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)
    
    def get_config(self):
        config = {'min': self.min, 'max': self.max}
        base_config = super(Clip, self).get_config()
        return dict(list(base_config.items()) + list(config.items() ))
    

global_layers_list['Clip']=Clip
global_layers_list['SortByFirstFeatureAndPool1D'] = SortByFirstFeatureAndPool1D


