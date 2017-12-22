import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool) #for batch normalization

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape,stride, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    # initialise weights and bias for the filter
    #weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    weights = tf.get_variable(name+'_W', conv_filt_shape, initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], padding='SAME')
    # add the bias
    out_layer += bias
    return out_layer

def wide_block1(inputs, n_input_plane, n_output_plane,stride1, stride2, i): #block with convolution in skip
    conv1 = create_new_conv_layer(inputs, n_input_plane, n_output_plane, [3, 3], stride1, name = 'conv'+str(i))
    conv1_bn = tf.layers.batch_normalization(conv1, training=is_training)
    conv1_relu = tf.nn.relu(conv1_bn)
    conv2 = create_new_conv_layer(conv1_relu, n_output_plane, n_output_plane, [3, 3], stride2, name = 'conv'+str(i+1))

    conv3 = create_new_conv_layer(inputs, n_input_plane, n_output_plane, [3, 3], stride1, name = 'conv'+str(i+2))
    output = conv2 + conv3
    return output

def wide_block2(inputs, n_input_plane, stride, i): #block without convolution in skip
    inputs_bn = tf.layers.batch_normalization(inputs, training=is_training)
    inputs_relu = tf.nn.relu(inputs_bn)
    conv1 = create_new_conv_layer(inputs_relu, n_input_plane, n_input_plane, [3, 3], stride, name = 'conv'+str(i))
    conv1_bn = tf.layers.batch_normalization(conv1, training=is_training)
    conv1_relu = tf.nn.relu(conv1_bn)
    conv2 = create_new_conv_layer(conv1_relu, n_input_plane, n_input_plane, [3, 3], stride, name = 'conv'+str(i+1))

    output = inputs + conv2
    return output

def between_blocks(inputs):
    inputs_bn = tf.layers.batch_normalization(inputs, training=is_training)
    inputs_relu = tf.nn.relu(inputs_bn)

    return inputs_relu

# before blocks
with tf.name_scope('before_blocks'):
    conv1 = create_new_conv_layer(x, 3, 16, [3, 3], 1, name = 'conv1')
    before_output = between_blocks(conv1)

#define blocks
# i defines number of first convolution in block
with tf.name_scope('blocks'):
    output = wide_block1(before_output, 16, 160, 1, 1, 2)
    output = wide_block2(output, 160, 1, 5)
    output = wide_block2(output, 160, 1, 7)
    output = wide_block2(output, 160, 1, 9)
    output = between_blocks(output)
    output = wide_block1(output, 160, 320, 2, 1, 11)
    output = wide_block2(output, 320, 1, 14)
    output = wide_block2(output, 320, 1, 16)
    output = wide_block2(output, 320, 1, 18)
    output = between_blocks(output)
    output = wide_block1(output, 320, 640, 2, 1, 20)
    output = wide_block2(output, 640, 1, 23)
    output = wide_block2(output, 640, 1, 25)
    output = wide_block2(output, 640, 1, 27)

#after blocks
with tf.name_scope('after_blocks'):
    final_output = between_blocks(output)
    ksize = [1, 8, 8, 1]
    strides = [1, 1, 1, 1]
    out_layer = tf.nn.avg_pool(final_output, ksize = ksize, strides = strides, padding = 'SAME', name = 'pooling')
    final_output = tf.reshape(out_layer, [-1, 640*8*8])

# layer with softmax activations
with tf.name_scope('softmax'):
    #wd = tf.Variable(tf.truncated_normal([640*8*8, 10], stddev=0.03), name = "softmax_W")
    #wd = tf.get_variable("dense_layer_W", [640*8*8, 10], initializer=tf.uniform_unit_scaling_initializer(factor=1))
    wd = tf.get_variable("dense_layer_W", [640*8*8, 10], initializer=tf.contrib.layers.xavier_initializer())
    bd = tf.Variable(tf.truncated_normal([10], stddev=0.01), name = "dense_layer_b")
    assert_op1 = tf.verify_tensor_all_finite(final_output, "final_output contains Nan or Inf")
    with tf.control_dependencies([assert_op1]):
        dense_layer = tf.matmul(final_output, wd) + bd
    dense_layer = dense_layer + 1e-7
    y_ = tf.nn.softmax(dense_layer)

lr_init = 0.1
lr_drop = 0.2
learning_rate = tf.Variable(initial_value=lr_init, dtype=tf.float32, trainable=False)
#learning rate definition
'''
def lr_def(epoch):
    if epoch < 60:
        learning_rate = lr_init
    elif epoch < 120:
        learning_rate = lr_init * lr_drop
    elif epoch < 160:
        learning_rate = lr_init * (lr_drop ** 2)
    else:
        learning_rate = lr_init * (lr_drop ** 3)
    return learning_rate
'''
def lr_def(epoch):
    if epoch < 3:
        learning_rate = lr_init
    elif epoch < 5:
        learning_rate = lr_init * lr_drop
    elif epoch < 20:
        learning_rate = lr_init * (lr_drop ** 2)
    else:
        learning_rate = lr_init * (lr_drop ** 3)
    return learning_rate


with tf.name_scope('loss'):
    #cross_entropy = -tf.reduce_sum(y*tf.log(y_ + 1e-7)) #to solve NaN problem with cross entropy
    assert_op = tf.verify_tensor_all_finite(dense_layer, "dense layer contains Nan or Inf")
    with tf.control_dependencies([assert_op]):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer, labels=y))
    all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    weight_decay = tf.reduce_mean([tf.reduce_mean(tf.square(w)) for w in all_weights]) #biases is also considered as weights
    total_loss = cross_entropy + 0.0005*weight_decay
with tf.name_scope('optimizer'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimiser = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(total_loss)
    #optimiser = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(total_loss)

# define an accuracy assessment operation
with tf.name_scope('accuracy'):
    accuracy = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", total_loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)

tf.summary.scalar("learning", learning_rate)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

'''
###count total number of trainable parameters
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)
'''