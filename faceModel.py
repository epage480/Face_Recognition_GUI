import tensorflow as tf
import propertyLib
import numpy as np


class faceModel:

    def __init__(self, x, y, is_training=True, keep_prob=.5, l_r=1e-4, l2_weight_decay=1e-4,trip_margin=.1, scope='face_scope'):
        with tf.variable_scope(scope, reuse=False):
            self.x = x
            self.y = y
            self.l_r = l_r
            self.l2_weight_decay = l2_weight_decay
            self.is_training = is_training
            self.keep_prob = keep_prob
            self.trip_margin = trip_margin
            self.scope = scope
            self.vars = None
            self.features = None
            self.inference
            self.ce_loss
            self.trip_loss
            self.loss
            self.optimize
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

    # Outputs unmodified logits i.e. without softmax
    @propertyLib.lazy_property
    def inference(self):
        """
        :return: # Network I originally used
        conv1 = tf.layers.conv2d(inputs=self.x, filters=24, kernel_size=9, strides=2, padding='same',
                                 activation=tf.nn.relu)
        print("conv1 shape:", conv1.get_shape())
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, padding='same')
        print("pool1 shape:", pool1.get_shape())
        conv2 = tf.layers.conv2d(inputs=pool1, filters=24, kernel_size=3, strides=1, padding='same',
                                 activation=tf.nn.relu)
        print("conv2 shape:", conv2.get_shape())
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=1, padding='same')
        print("pool2 shape:", pool2.get_shape())
        pool2 = tf.contrib.layers.flatten(pool2)
        fc1 = tf.contrib.layers.fully_connected(pool2, 256, tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256, tf.nn.relu)
        """

        def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, name, strides=(1,1,1,1),activation=tf.nn.relu):
            # setup the filter input shape for tf.nn.conv_2d
            conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

            # initialise weights and bias for the filter
            weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name + '_W')
            bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

            # setup the convolutional layer operation
            out_layer = tf.nn.conv2d(input_data, weights, strides, padding='SAME')

            # add the bias
            out_layer += bias

            # apply a ReLU non-linear activation
            return  activation(out_layer)

        # conv1
        conv1 = create_new_conv_layer(self.x, 3, 64, (7,7,3), "conv1", strides=[1,2,2,1])
        print("conv1 shape:", conv1.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("# of parameters:", params)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        print("pool1 shape:", pool1.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])-params
        print("# of parameters:", params)

        # conv2a
        conv2a = create_new_conv_layer(pool1, 64, 64, (1, 1, 64), "conv2a")
        print("conv2a shape:", conv2a.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv2a
        conv2 = create_new_conv_layer(conv2a, 64, 192, (3, 3, 64), "conv2")
        print("conv2 shape:", conv2.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')
        print("pool2 shape:", pool2.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv3a
        conv3a = create_new_conv_layer(pool2, 192, 192, (1, 1, 192), "conv3a")
        print("conv3a shape:", conv3a.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv3
        conv3 = create_new_conv_layer(conv3a, 192, 384, (3, 3, 384), "conv3")
        print("conv3 shape:", conv3.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # pool3
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
        print("pool3 shape:", pool3.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv4a
        conv4a = create_new_conv_layer(pool3, 384, 384, (1, 1, 384), "conv4a")
        print("conv4a shape:", conv4a.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv4
        conv4 = create_new_conv_layer(conv4a, 384, 256, (3, 3, 256), "conv4")
        print("conv4 shape:", conv4.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv5a
        conv5a = create_new_conv_layer(conv4, 256, 256, (1, 1, 256), "conv5a")
        print("conv5a shape:", conv5a.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv5
        conv5 = create_new_conv_layer(conv5a, 256, 256, (3, 3, 256), "conv5")
        print("conv5 shape:", conv5.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv6a
        conv6a = create_new_conv_layer(conv5, 256, 256, (1, 1, 256), "conv6a")
        print("conv6a shape:", conv6a.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # conv6
        conv6 = create_new_conv_layer(conv6a, 256, 256, (3, 3, 256), "conv6")
        print("conv6 shape:", conv6.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # pool4
        pool4 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        print("pool4 shape:", pool4.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        # concat
        concat = tf.contrib.layers.flatten(pool4)
        print("concat shape:", concat.get_shape())
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        fc1 = tf.contrib.layers.fully_connected(concat, 4096, tf.nn.relu)
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)
        fc2 = tf.contrib.layers.fully_connected(fc1, 4096, tf.nn.relu)
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) - params
        print("# of parameters:", params)

        L2 = tf.contrib.layers.fully_connected(fc1, 128, tf.nn.relu)

        return L2

    @propertyLib.lazy_property
    def trip_loss(self):
        return tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=self.y, embeddings=self.inference, margin=self.trip_margin)

    @propertyLib.lazy_property
    def ce_loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inference, labels=self.y))

    # L2 Loss
    @propertyLib.lazy_property
    def l2_loss(self):
        # Calculate l2 loss by collecting all trainable variables and summing their individual losses
        tmp = []
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        for var in self.vars:
            tmp.append(tf.reshape(var, [-1]))
        tmp = tf.concat(tmp, 0)
        print("# of Initial Parameters:", tmp.shape)
        l2_loss = 0
        for var in self.vars:
            l2_loss += tf.nn.l2_loss(var)
        return l2_loss * self.l2_weight_decay

    @propertyLib.lazy_property
    def loss(self):
        return self.trip_loss

    # Optimize based on both cross entropy and l2 loss
    @propertyLib.lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self.l_r).minimize(self.loss)
        #self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        #return tf.train.AdamOptimizer(self.l_r).minimize(self.loss, var_list=self.vars)

    @propertyLib.lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.y, 1), tf.argmax(self.inference, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @propertyLib.lazy_property
    def accuracy(self):
        prediction = tf.argmax(self.inference, 1)
        num_equal = tf.equal(prediction, tf.argmax(self.y, 1))
        return tf.reduce_mean(tf.cast(num_equal, tf.float32))

    def save(self, sess, model_path):
        return self.saver.save(sess, model_path)

    def restore(self, sess, model_path):
        return self.saver.restore(sess, model_path)
