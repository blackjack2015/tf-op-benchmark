from __future__ import print_function
import tensorflow as tf
import numpy as np
import time

config = tf.ConfigProto(device_count={"CPU": 1}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 1, 
                intra_op_parallelism_threads = 12,
                log_device_placement=True)

inputs = [[16, 224, 224, 3], [16, 112, 112, 64], [16, 56, 56, 128]]
filters = [[7, 7, 3, 64], [3, 3, 64, 128], [3, 3, 128, 256]]
strides = [[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
batch_sizes = [32, 64, 128, 256]

log_info = ""
for idx in range(len(inputs)):
    for bs in batch_sizes:
        input_shape = inputs[idx]
        input_shape[0] = bs
        filter_shape = filters[idx]
        stride = strides[idx]
        input = tf.Variable(tf.random_normal(input_shape))
        filter = tf.Variable(tf.random_normal(filter_shape))
        z = tf.nn.conv2d(input, filter, strides=stride, padding="VALID")
        iters = 100
        log_info += "%s, %s, %s, %s: " % (bs, input_shape, filter_shape, stride)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
        
            #CONV
            for i in range(-2,iters):
                if i == 0:
                    t = time.time()
                sess.run(z)

            total_time = time.time() - t
        
            log_info += "%f. \n" % (total_time/iters)

print(log_info)
print('CONV benchmark over')
