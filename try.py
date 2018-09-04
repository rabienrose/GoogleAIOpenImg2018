import tensorflow as tf
cond_value = tf.Variable(False)
data=tf.constant([0])
def func1():
    data1 = tf.Print(data, [data[1]], message='func1')
    return data1

def func2():
    data1=tf.Print(data,[data[0]],message='func2')
    return data1

cond_result = tf.cond(cond_value, func1, func2)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(cond_result))


# import tensorflow as tf
# cond_value = tf.Variable(5)
# data=tf.constant([1,2])
# cond_result = tf.cond(tf.greater(cond_value, 1), lambda: data[cond_value-1], lambda: data[cond_value])
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(cond_result))

# import tensorflow as tf
#
# X1 = tf.Variable(1.)
# X2 = tf.Variable(1.)
#
# def fun1():
#     ass1=tf.assign(X1, 2.)
#     ass1 = tf.Print(ass1, [ass1], message='x1')
#     return ass1
#
# def fun2():
#     ass2=tf.assign(X2, 2.)
#     ass2=tf.Print(ass2,[ass2], message='x2')
#     return ass2
#
#
# cond_value = tf.Variable(True)
# cond_result = tf.cond(cond_value, fun1, fun2)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     sess.run(cond_result)
#     print(sess.run(X1), sess.run(X2))
