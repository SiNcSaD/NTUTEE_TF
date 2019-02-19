import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W")
biases = tf.Variable(tf.zeros([1]), name="B")

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer() 

## 可視化資料
# plt.plot(x_data, y_data, 'o')
# plt.show()

with tf.Session() as sess:
    sess.run(init)          # Very important

    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))