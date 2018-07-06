import tensorflow as tf
from keras.engine.topology import Layer, InputSpec

class _GlobalTrimmedAveragePool(Layer):
    def __init__(self, **kwargs):
        super(_GlobalTrimmedAveragePool, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],1)
    def call(self, inputs):
        raise NotImplementedError

class TrimmedAveragePool(_GlobalTrimmedAveragePool):
    def call(self, inputs):
        inputs=tf.transpose(inputs,[0,2,1])
        top= tf.nn.top_k(inputs,k=5)[0]#Dinamic average pooling k=5
        P=tf.reduce_mean(top,axis=-1)
        return P
