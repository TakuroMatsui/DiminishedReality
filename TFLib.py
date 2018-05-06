import tensorflow as tf

class TFLib:
    def __init__(self):
        self.config = tf.ConfigProto(#sess(config=...)
            gpu_options=tf.GPUOptions(
                visible_device_list="0", # specify GPU number
                allow_growth=True
            )
        )

    def fc_variable(self,weight_shape,name="fc"):
        with tf.variable_scope(name):
            weight_shape=(int(weight_shape[0]),int(weight_shape[1]))
            weight=tf.get_variable("w",weight_shape,initializer=tf.contrib.layers.xavier_initializer())
            bias=tf.get_variable("b",[weight_shape[1]],initializer=tf.constant_initializer(0.1))
        return weight,bias

    def conv_variable(self,weight_shape,name="conv"):
        with tf.variable_scope(name):
            weight_shape=(int(weight_shape[0]),int(weight_shape[1]),int(weight_shape[2]),int(weight_shape[3]))
            weight = tf.get_variable("w",weight_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias=tf.get_variable("b",[weight_shape[3]],initializer=tf.constant_initializer(0.1))
        return weight,bias

    def deconv_variable(self,weight_shape,name="deconve"):
        with tf.variable_scope(name):
            weight_shape=(int(weight_shape[0]),int(weight_shape[1]),int(weight_shape[2]),int(weight_shape[3]))
            weight = tf.get_variable("w",weight_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias=tf.get_variable("b",[weight_shape[2]],initializer=tf.constant_initializer(0.1))
        return weight,bias

    def conv2d(self,x,w,stride=1):
        return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding="SAME")

    def deconv2d(self,x,w,output_shape,stride=1):
        output_shape=(int(output_shape[0]),int(output_shape[1]),int(output_shape[2]),int(output_shape[3]))
        return tf.nn.conv2d_transpose(x,w,output_shape=output_shape,strides=[1,stride,stride,1],padding="SAME")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x)

    def maxpool(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    