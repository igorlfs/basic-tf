import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# |%%--%%| <RpB5djYN3C|kipCOFwVnT>

tf.constant(4)

# |%%--%%| <kipCOFwVnT|NGkgwTAqjl>

tf.constant(4, shape=(1, 1), dtype=tf.float32)

# |%%--%%| <NGkgwTAqjl|iGHRjrrIYt>

# rank-1
tf.constant([1, 2, 3])

# |%%--%%| <iGHRjrrIYt|LvRsbxugah>

# rank-2
tf.constant([[1, 2, 3], [4, 5, 6]])

# |%%--%%| <LvRsbxugah|N5Vu8iCXCs>

tf.ones((3, 3))

# |%%--%%| <N5Vu8iCXCs|0VqiNyznJm>

tf.zeros((3, 3))

# |%%--%%| <0VqiNyznJm|GOKF4Ute5t>

tf.eye(3)

# |%%--%%| <GOKF4Ute5t|QSyBs7qnJq>

tf.random.normal((3, 3), mean=0, stddev=1)

# |%%--%%| <QSyBs7qnJq|kep0QDa2k9>

tf.random.uniform((3, 3), minval=0, maxval=1)

# |%%--%%| <kep0QDa2k9|RM1nd2dMwD>

x = tf.range(10)
print(x)
x = tf.cast(x, dtype=tf.float32)
print(x)

# |%%--%%| <RM1nd2dMwD|03UGXS0MgS>

x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])

z = tf.add(x, y)
z = x + y
print(z)
z = tf.subtract(x, y)
z = x - y
print(z)
z = tf.divide(x, y)
z = x / y
print(z)
z = tf.multiply(x, y)
z = x * y
print(z)
z = tf.tensordot(x, y, axes=1)
print(z)
z = x**3
print(z)

# |%%--%%| <03UGXS0MgS|wNL0yBuG29>

x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))

z = tf.matmul(x, y)
z = x @ y
print(z)

# |%%--%%| <wNL0yBuG29|95Sz82Etmf>

# sliing, indexing
x = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
print(x[0, :])
print(x[:, 0])
print(x[0, 1:3])
print(x[0, 1])
# |%%--%%| <95Sz82Etmf|RMMGOZx460>

# reshaping
x = tf.random.normal((2, 3))
print(x)
x = tf.reshape(x, (3, 2))
print(x)
x = tf.reshape(x, (-1, 2))
print(x)

# |%%--%%| <RMMGOZx460|Onpl8IMm0T>

# numpy
x = tf.random.normal((2, 3))
print(x)
x = x.numpy()
print(type(x))
x = tf.convert_to_tensor(x)
print(type(x))

# |%%--%%| <Onpl8IMm0T|3tvecTSi4w>

tf.constant(["Lula", "Lorem", "Ipsum"])
