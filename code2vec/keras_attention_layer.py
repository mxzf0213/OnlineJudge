import tensorflow as tf

try:
    import tensorflow.python.keras as keras
    from tensorflow.python.keras import layers
    from tensorflow.python.keras import backend as K
except:
    import tensorflow.keras as keras
    from tensorflow.keras import layers
    from tensorflow.keras import backend as K

class Attention_layer(layers.Layer):
    def __init__(self,**kwargs):
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, inputs_shape):
        assert isinstance(inputs_shape, list)

        # inputs: --> size
        #     [(None,max_contents,code_vector_size),(None,max_contents)]
        #     the second input is optional
        if(len(inputs_shape)<1 or len(inputs_shape)>2):
            raise ValueError("AttentionLayer expect one or two inputs.")

        # (None,max_contents,code_vector_size)
        input_shape = inputs_shape[0]

        if(len(input_shape)!=3):
            raise ValueError("Input shape for AttentionLayer shoud be of 3 dimensions.")

        self.input_length = int(input_shape[1])
        self.input_dim = int(input_shape[2])

        attention_param_shape = (self.input_dim, 1)

        self.attention_param = self.add_weight(
            name = 'attention_param',
            shape = attention_param_shape,
            initializer = 'uniform',
            trainable = True,
            dtype = tf.float32
        )

        super(Attention_layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert  isinstance(inputs, list)

        # inputs:
        #     [(None,max_contents,code_vector_size),(None,max_contents)]
        #     the second input is optional
        if (len(inputs) < 1 or len(inputs) > 2):
            raise ValueError("AttentionLayer expect one or two inputs.")

        actual_input = inputs[0]
        mask = inputs[1] if(len(inputs)>1) else None

        if mask is not None and not(((len(mask.shape)==3 and mask.shape[2] == 1) or (len(mask.shape) == 2)) and (mask.shape[1] == self.input_length)):
            raise ValueError("`mask` shoud be of shape (batch, input_length) or (batch, input_length, 1) when calling AttentionLayer.")

        assert actual_input.shape[-1] == self.attention_param.shape[0]

        # (batch, input_length, input_dim) * (input_dim, 1) ==> (batch, input_length, 1)
        attention_weights = K.dot(actual_input, self.attention_param)

        if mask is not None:
            if(len(mask.shape) == 2):
                mask = K.expand_dims(mask, axis=2)  #(batch, input_dim, 1)
            mask = K.log(mask)  #e.g. K.exp(K.log(0.)) = 0 K.exp(K.log(1.)) =1
            attention_weights += mask

        attention_weights = K.softmax(attention_weights, axis=1)
        result = K.sum(actual_input * attention_weights, axis=1)
        return result, attention_weights

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]   #(batch,input_length,input_dim) --> (batch,input_dim)