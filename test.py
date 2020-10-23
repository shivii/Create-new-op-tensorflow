import tensorflow as tf

add_one_module = tf.load_op_library('/content/new_op/add_one.so')

#my_tensor = tf.constant([[1, 2], [3, 4]])
#my_variable = tf.Variable(my_tensor)

my_placeholder = tf.placeholder(tf.int32, (2,2))
result = add_one_module.add_one(my_placeholder)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  result = sess.run(result, feed_dict={my_placeholder:[[1, 2], [3, 4]]})

print(result)