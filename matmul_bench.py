import os
import sys
import tensorflow as tf
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, help='matrix size, N x N', default=256)

opt = parser.parse_args()
n = opt.N
dtype = tf.float32
with tf.device("/cpu:0"):
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)

inter_op_parallelism = 4
#if "OMP_NUM_THREADS" in os.environ and int(os.environ["OMP_NUM_THREADS"]) > 12:
#    inter_op_parallelism = 2
#    os.environ["OMP_NUM_THREADS"] = str(int((int(os.environ["OMP_NUM_THREADS"]) / 2)))

# avoid optimizing away redundant nodes
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)), device_count={"CPU": 1},
                inter_op_parallelism_threads = inter_op_parallelism, 
                intra_op_parallelism_threads = 24,
                log_device_placement=True)
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
iters = 100

# pre-warming
sess.run(product.op)

for i in range(-2, iters):
  if i == 0:
    start = time.time()
  sess.run(product.op)
end = time.time()
ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed = (end - start)
rate = iters*ops/elapsed/10**9
print('\n %d x %d matmul took: %.2f msec, %.2f G ops/sec' % (n, n,
                                                            elapsed/iters*1000.0,
                                                            rate,))
