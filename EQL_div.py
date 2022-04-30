import tensorflow as tf
import keras

from EQL_div.inputs_needed import inputs_needed


class EQL_div_network(keras.Model):
    def __init__(self, funcs, l1_reg=0., l0_thresh=0., penalty_strength=1., eval_bound=10., init_stddev=0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.funcs = funcs  # function types (list of lists of strings)
        self.penalty_strength = tf.constant(penalty_strength, dtype=tf.float64)  # strength of the reg_div penalty term
        # L1-regularization strength (\lambda in paper)
        self.l1_reg = tf.Variable(l1_reg, trainable=False, dtype=tf.float64)
        self.l0_thresh = tf.Variable(l0_thresh, trainable=False, dtype=tf.float64)  # L0-threshold
        self.init_stddev = init_stddev # stddev for RandomNormal initializer
        self.nr_layers = len(funcs)  # number of EQL_div layers

        # store the number of dense nodes in each layer
        self.dense_nodes = [sum(inputs_needed(func) for func in funcs[i]) for i in range(self.nr_layers)]

        # store the step we're at (the number of times call() was called with training=True)
        self.step = tf.Variable(0., trainable=False, dtype=tf.float64)

        # switch to indicate if we're in a training or a penalty epoch (0.0 is off, 1.0 is on)
        self.penalty_epoch = tf.Variable(0.0, trainable=False)

        # boundary to values in the evaluation range during penalty epochs
        self.eval_bound = eval_bound

        # store a dictionary with the possible functions (strings) and their corresponding tf functions
        self.func_dict = {'id': tf.identity,
                          'sin': tf.sin,
                          'cos': tf.cos,
                          'prod': tf.multiply,
                          'square': tf.square,
                          'sqrt': self.reg_sqrt,
                          'div': self.reg_div}

    def set_l1_reg(self, l1_reg):
        # using this method, we can change the regularization strength without recompiling the model
        self.l1_reg.assign(l1_reg)
        return

    def set_l0_thresh(self, l0_thresh):
        # using this method, we can change the L0 threshold without recompiling the model
        self.l0_thresh.assign(l0_thresh)
        return

    def set_penalty_epoch(self, penalty_epoch):
        # method to change self.penalty_epoch
        if penalty_epoch:
            self.penalty_epoch.assign(1.0)
        else:
            self.penalty_epoch.assign(0.0)
        return

    def reg_div(self, numerator, denominator):
        # perfom the regularized division and add the penalty term to the losses

        # get the current division threshold \theta(t)
        div_thresh = 0.01 / tf.sqrt(self.step + 1)
        # 1. if normal division needed, 0. if regularization in effect
        mask = tf.cast(denominator > div_thresh, dtype=tf.float64)
        # calculate the output
        output = mask * numerator * tf.math.reciprocal(tf.abs(denominator) + 1e-10)

        # add the reg_div penalty term (in both cases of penalty epoch and normal training epoch)
        self.add_loss(self.penalty_strength * tf.reduce_sum((1. - mask) * (div_thresh - denominator)))

        # add another loss that tries to keep denominators away from 0 (above 0.1)
        min_denom = 0.1
        mask = tf.cast(denominator > min_denom, dtype=tf.float64)
        self.add_loss(10*self.l1_reg * tf.reduce_sum((1. - mask) * (div_thresh - denominator)))

        return output

    def reg_sqrt(self, nr):
        # perfom the regularized square root and add the penalty term to the losses

        # get the current sqrt threshold
        sqrt_thresh = 0.01 / tf.sqrt(self.step + 1)
        mask = tf.cast(nr > sqrt_thresh, dtype=tf.float64)
        # calculate the output
        output = mask * tf.math.sqrt(tf.math.abs(nr))

        # add the reg_div penalty term (in both cases of penalty epoch and normal training epoch)
        self.add_loss(self.penalty_strength * tf.reduce_sum((mask-1.) * nr))

        return output

    def build(self, input_shape):
        # get all the weights and biases

        # w and b are the weights and biases of the dense layers
        self.w = list()
        self.b = list()

        # first layer is special
        self.w.append(self.add_weight(shape=(input_shape[-1], self.dense_nodes[0]),
                                      trainable=True, dtype=tf.float64,
                                      initializer=tf.keras.initializers.RandomNormal(stddev=self.init_stddev)))
        self.b.append(self.add_weight(shape=(self.dense_nodes[0],),
                                      trainable=True, dtype=tf.float64,
                                      initializer=tf.keras.initializers.RandomNormal(stddev=self.init_stddev)))

        for i in range(1, self.nr_layers):
            self.w.append(self.add_weight(shape=(len(self.funcs[i - 1]), self.dense_nodes[i]),
                                          trainable=True, dtype=tf.float64,
                                          initializer=tf.keras.initializers.RandomNormal(stddev=self.init_stddev)))
            self.b.append(self.add_weight(shape=(self.dense_nodes[i],),
                                          trainable=True, dtype=tf.float64,
                                          initializer=tf.keras.initializers.RandomNormal(stddev=self.init_stddev)))

        # initialize denominators to 1
        for i in range(self.nr_layers):
            ws = tf.unstack(self.w[i], axis=1)
            bs = tf.unstack(self.b[i], axis=0)

            at_node = 0
            for j in range(len(self.funcs[i])):
                if self.funcs[i][j] == 'div':
                    ws[at_node+1] = 0. * ws[at_node+1]
                    bs[at_node+1] = 0. * bs[at_node+1] + 1.
                if self.funcs[i][j] == 'sqrt':
                    ws[at_node] = tf.abs(ws[at_node])
                    bs[at_node] = tf.abs(bs[at_node])
                at_node += inputs_needed(self.funcs[i][j])
            self.w[i].assign(tf.stack(ws, axis=1))
            self.b[i].assign(tf.stack(bs, axis=0))

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # we don't need x
        del x

        def normal_loss(y, y_pred, regularization_losses):
            res = tf.reduce_mean(tf.math.squared_difference(tf.cast(y, tf.float64), tf.cast(y_pred, dtype=tf.float64)))
            res = tf.math.log(1. + res)
            res += tf.reduce_sum(regularization_losses)
            return res
        # get the mse (training epoch) or penalty epoch term into the loss
        loss = tf.cond(tf.math.equal(self.penalty_epoch, 0.0),
                       lambda: normal_loss(y, y_pred, self.losses),
                       lambda: self.penalty_strength * tf.reduce_sum(tf.maximum(y_pred - self.eval_bound, 0)
                                                                     + tf.maximum(- y_pred - self.eval_bound, 0)))
        return loss

    def enforce_zeros(self):
        # enforce L0-Norm of all weights
        for i in range(self.nr_layers):
            # treat w of current layer
            zeros_w = tf.cast(tf.abs(self.w[i]) > tf.fill(self.w[i].shape, self.l0_thresh),
                              dtype=self.w[i].dtype)
            self.w[i].assign(tf.multiply(self.w[i], zeros_w))

            # treat b of current layer
            zeros_b = tf.cast(tf.abs(self.b[i]) > tf.fill(self.b[i].shape, self.l0_thresh),
                              dtype=self.b[i].dtype)
            self.b[i].assign(tf.multiply(self.b[i], zeros_b))
        return

    def train_step(self, data):

        # copy-paste of standard train_step, DO NOT CHANGE
        x, y, sample_weight = keras.engine.data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # end of copy-paste of standard train_step, DO NOT CHANGE

        self.enforce_zeros()

        return self.compute_metrics(x, y, y_pred, sample_weight)

    def call(self, inputs, training=False, **kwargs):
        if training:
            self.step.assign_add(1.)

        outputs = tf.cast(inputs, dtype=tf.float64)
        # go through all the layers, compute the result, and get the L1-regularization losses
        for i in range(self.nr_layers):
            # apply the dense layer
            outputs = tf.matmul(outputs, self.w[i]) + self.b[i]

            # apply the functions
            # get the number of inputs needed for any of the operations
            indices = [inputs_needed(func) for func in self.funcs[i]]
            # split the dense_output such that each slice can be processed by one of the functions
            slices = tf.split(outputs, indices, axis=1)
            # get a list of the functions (the actual tf functions) in the current layer
            curr_funcs = [self.func_dict[func] for func in self.funcs[i]]
            # get a list of all the individual outputs
            outputs = [func(*tf.unstack(output_slice, axis=1))
                       for func, output_slice in zip(curr_funcs, slices)]
            # stack the outputs together to get a tensor again
            outputs = tf.stack(outputs, axis=1)

            # add L1-regularization loss (if we are not in a penalty epoch)
            self.add_loss(self.l1_reg * tf.reduce_sum(tf.abs(self.w[i])))
            self.add_loss(self.l1_reg * tf.reduce_sum(tf.abs(self.b[i])))

        return outputs
