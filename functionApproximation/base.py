#!/usr/bin/python
# encoding: utf-8


import numpy as np
import ipdb
import tensorflow as tf
import logger

class basic_model(object):
    GRAPHS = {}
    SESS = {}
    SAVER = {}

    def c_opt(self,learning_rate,name):
        if str(name).__contains__("adam"):
            print("adam")
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif str(name).__contains__("adagrad"):
            print("adagrad")
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif str(name).__contains__("sgd"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif str(name).__contains__("rms"):
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif str(name).__contains__("moment"):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.1)
        return optimizer

    @classmethod
    def create_model(cls, config, variable_scope = "target", trainable = True, graph_name="DEFAULT",task_index=0):
        jobs = config.jobs
        job = list(jobs.keys())[0]
        logger.info("CREATE MODEL", config.model, "GRAPH", graph_name, "VARIABLE SCOPE", variable_scope,"jobs",jobs,"job",job,"task_index",task_index)
        cls.CLUSTER = tf.train.ClusterSpec(jobs)
        cls.SERVER = tf.train.Server(cls.CLUSTER, job_name=job, task_index=task_index,config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        if not graph_name in cls.GRAPHS:
            logger.info("Adding a new tensorflow graph:",graph_name)
            cls.GRAPHS[graph_name] = tf.Graph()
        with cls.GRAPHS[graph_name].as_default():
            model = cls(config, variable_scope=variable_scope, trainable=trainable)
            if not graph_name in cls.SESS:
                cls.SESS[graph_name] = tf.Session(cls.SERVER.target)
                cls.SAVER[graph_name] = tf.train.Saver(max_to_keep=50)
            cls.SESS[graph_name].run(model.init)
        return {"graph": cls.GRAPHS[graph_name],
               "sess": cls.SESS[graph_name],
               "saver": cls.SAVER[graph_name],
               "model": model,"cluster":cls.CLUSTER,"server":cls.SERVER}

    @classmethod
    def create_model_without_distributed(cls, config, variable_scope = "target", trainable = True, graph_name="DEFAULT"):
        logger.info("CREATE MODEL", config.model, "GRAPH", graph_name, "VARIABLE SCOPE", variable_scope)
        if not graph_name in cls.GRAPHS:
            logger.info("Adding a new tensorflow graph:",graph_name)
            cls.GRAPHS[graph_name] = tf.Graph()
        with cls.GRAPHS[graph_name].as_default():
            model = cls(config, variable_scope=variable_scope, trainable=trainable)
            if not graph_name in cls.SESS:
                cls.SESS[graph_name] = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
                cls.SAVER[graph_name] = tf.train.Saver(max_to_keep=50)
            cls.SESS[graph_name].run(model.init)
        return {"graph": cls.GRAPHS[graph_name],
               "sess": cls.SESS[graph_name],
               "saver": cls.SAVER[graph_name],
               "model": model}

    def _update_placehoders(self):
        self.placeholders = {"none":{}}
        raise NotImplemented

    def _get_feed_dict(self,task,data_dicts):
        place_holders = self.placeholders[task]
        res = {}
        for key, value in place_holders.items():
            res[value] = data_dicts[key]
        return res

    def __init__(self, args, variable_scope = "target", trainable = True):
        print(self.__class__)
        self.args = args
        self.variable_scope = variable_scope
        self.trainable = trainable
        self.placeholders = {}
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self.variable_scope):
            self._create_placeholders()
            self._create_global_step()
            self._update_placehoders()
            self._create_inference()
            if self.trainable:
                self._create_optimizer()
            self._create_intializer()

    def _create_global_step(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_intializer(self):
        with tf.name_scope("initlializer"):
            self.init = tf.global_variables_initializer()

    def _create_placeholders(self):
        raise NotImplementedError

    def _create_inference(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def chose_action(self, state, sess):
        raise NotImplementedError
        pass

    def build_cell(self,rnn_type,initializer,hidden,input_data,initial_state):
        cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.GRUCell(hidden,
                                                                   kernel_initializer=initializer,
                                                                   bias_initializer=tf.zeros_initializer)] * self.args.rnn_layer)
        return tf.nn.dynamic_rnn(cell, input_data, initial_state=initial_state,
                                 dtype=tf.float32,
                                 time_major=True)


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs
