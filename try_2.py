import tensorflow as tf

aa=tf.Variable(1)
bb=aa*2
#cc=tf.assign(bb1,5)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(bb))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(bb))
