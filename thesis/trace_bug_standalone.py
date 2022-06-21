# See https://github.com/tensorflow/profiler/issues/401
import tensorflow as tf


class FakeModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(16, activation=tf.nn.relu, name="bar")
        self.l2 = tf.keras.layers.Dense(8, name="baz")

    def call(self, x, *args, **kwargs):
        y = self.l1(x)
        y = self.l2(y)

        return y


fake_model = FakeModel()


with tf.profiler.experimental.Profile("logdir"):
    for i in range(10):
        fake_batch = tf.random.normal((8, 32))
        with tf.profiler.experimental.Trace("test", step_num=i):
            fake_model(fake_batch)
