# Python_TensorFlow_Basics

=Python =

==Python 模块说明==

>>> help("模块名")


==文件目录和文件名操作==

#基本函数
 import os

 os.listdir()
 os.path.join()
 os.path.basename()

# current file path

 import os
 os.path.realpath('.')

#字符串类型的split()方法，通过指定分隔符对字符串进行分割

 str=('24_0.png')

 str1=str.split('.')[0]

 str2=str1.split('_')

 print(str2[1])

#字符串str转换成int: int_value = int(str_value)

 a=int(str2[1])

#glob 模块，用于逐个获取匹配文件路径名

 import glob

 glob.glob(r'./*.py')

 f=glob.iglob(r"./*.*")

 for py in f:
   print(py)

==IPython==

* IPython 

命令行输入:

 $ ipython notebook

或者使用：

 $jupyter notebook

=Python Math library=

https://docs.python.org/3.6/library/math.html

==Numpy ==

[x] python-numpy, http://cs231n.github.io/python-numpy-tutorial/.

==matplotlib==
（matplotlib/pyplot）展示WaveForm  AudioPlot

<code>
 import matplotlib.pyplot as plt
 from wavReader import readWav

 rate, data =readWav('c:\\Users\\icenter\\Documents\\a.wav')

 plt.plot(data)

 plt.show()

</code>

=Activation Function（激活函数）=

==ReLU函数 ==

ReLU函数是Softplus函数的钝化版本。ReLU函数有几种推广版本：

绝对值整流（absolute value rectification）。

渗漏整流线性单元（Leaky ReLU）。

<code>
 import matplotlib.pyplot as plt
 import numpy as np

 x = np.linspace(-5,5,100)

 relu = lambda x: np.maximum(x, 0)

 plt.plot(x, relu(x), color='blue', lw=2)

 plt.show()
</code>

==PReLU==

参数化整流线性单元（parametric ReLU）或者PReLU。

<code>
 import matplotlib.pyplot as plt
 import numpy as np
 
 x = np.linspace(-5,5,100)
 pr = lambda xs: [x*0.1 if x <0 else x for x in xs] 
 
 plt.plot(x, pr(x), color='blue', lw=2)
 plt.show()
</code>

==Softplus函数 ==
Softplus函数是ReLu函数的软化版本。

<code>
 import matplotlib.pyplot as plt
 import numpy as np

 x = np.linspace(-5,5,100)

 softplus = lambda x: np.log(1.0 + np.exp(x))

 plt.plot(x, softplus(x), color='blue', lw=2)

 plt.show()
</code>

== sigmoid函数 ==
（http://matplotlib.org/）

 import matplotlib.pyplot as plt
 import numpy as np

 x = np.linspace(-5,5,100)

 sigmoid = lambda x: 1 / (1 + np.exp(-x))

 plt.plot(x,sigmoid(x), color='red', lw=2)

 plt.show()

==饱和型S函数==
saturated function

<code>
 import matplotlib.pyplot as plt
 import numpy as np

 x = np.linspace(-5,5,100)

 pr = lambda xs: [0 if x <0 else 1 for x in xs] 

 plt.plot(x, pr(x), color='blue', lw=2)

 plt.show()
</code>

==logit函数 ==
分对数

<code>
 import matplotlib.pyplot as plt
 import numpy as np

 x = np.linspace(-5,5,100)

 logit = lambda x: np.log(x / (1 - x))

 plt.plot(x,logit(x), color='blue', lw=2)

 plt.show()
</code>

==SoftSign函数==



<code>
 import matplotlib.pyplot as plt
 import numpy as np

 x = np.linspace(-5,5,100)

 softsign = lambda x: x / (1 + np.absolute(x))

 plt.plot(x,softsign(x), color='blue', lw=2)

 plt.show()
</code>

==tanh函数==

<code>
 import matplotlib.pyplot as plt
 import numpy as np

 x = np.linspace(-5,5,100)

 th = lambda x: np.tanh(x)

 plt.plot(x,th(x), color='blue', lw=2)

 plt.show()
</code>


==hard tanh函数==

<code>
 import matplotlib.pyplot as plt
 import numpy as np

 def hard_tanh(x):
     if abs(x)<1:
         return x
     else:
         if x>1: 
             return 1
         else:
             return -1

 x = np.linspace(-5,5,100)

 htan = lambda xs: [hard_tanh(x) for x in xs] 

 plt.plot(x,htan(x), color='blue', lw=2)

 plt.show()

</code>

==softmax函数 ==

 import math
 import matplotlib.pyplot as plt

 w=[1,2,3,4,5,6,7,8,9]

 w_exp=[math.exp(i) for i in w]

 print(w_exp)

 sum_w_exp = sum(w_exp)

 softmax = [round(i / sum_w_exp, 3) for i in w_exp]

 print(softmax)

 print(sum(softmax))

 plt.plot(softmax)

 plt.show()

----

=Tensorflow=

TensorFlow：opensource machine intelligence libraries

==TensorFlow原理==

TensorFlow是一个用数据流图进行数值计算的软件库。图中的节点表示的数学运算，而图的边代表它们之间传送的多维数据阵列（张量）。

Tensor（张量）意味着N维数组。Tensor的1维形式是向量，2维是矩阵；图像可以用三维Tensor（行，列，颜色）来表示。

TensorFlow用于模型训练过程的数据流图，包括训练数据的读取和转换，队列，参数的更新以及周期性监测点生成。计算图中的操作都是并发执行的，图中的节点的可变状态（Mutable states）在图的执行中是可以共享的。

==TensorFlow本质==

TensorFlow是一种元编程（meta programming），构建计算图的语言。

TensorFlow求导采用符号微分方法（Symbolic differentiation）。

神经网络搭建，可以很容易的通过TensorFlow的计算图的完成。TensorFlow生成自动求导的计算图。

TensorFlow参考了Python中的Numpy库的很多概念和函数，如Arrays的概念、Shape的概念、reduce_sum()函数，reshape()函数，argmax()函数等等。

TensorFlow中的Tensor可以理解为就是Numpy中Array的概念。当然TensorFlow和Numpy的定位是不同的。

==TensorFlow的变量和运算 ==

<code>
 import tensorflow as tf
 import numpy

 A=tf.Variable([[1,2],[3,4]], dtype=tf.float32)
 A.get_shape()

TensorShape([Dimension(2), Dimension(2)])

 B=tf.Variable([[5,6],[7,8]],dtype=tf.float32)
 B.get_shape()

TensorShape([Dimension(2), Dimension(2)])

 C=tf.matmul(A,B)

 tf.global_variables()
[<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(2, 2) dtype=float32_ref>]

 init = tf.global_variables_initializer()

 sess =tf.Session()

 sess.run(init)
</code>

 print(sess.run(C))
[[ 19.  22.]
[ 43.  50.]]

 print(sess.run(tf.reduce_sum(C, 0)))
[ 62.  72.]

 print(sess.run(tf.reduce_sum(C, 1)))
[ 41.  93.]

=TensorFlow 的表示= 

==Sigmoid 函数/SOFTMAX 函数/SOFTPLUS函数==

（https://www.tensorflow.org/api_guides/python/nn）

 R=tf.Variable([1.,.2,3],dtype=tf.float32)

 T=tf.Variable([True, True, False, False], dtype=tf.bool)

 U=tf.Variable([True, True, True, False], dtype=tf.bool)

 init=tf.global_variables_initializer()

 sess.run(init) //sess.run(tf.global_variables_initializer())

 print(sess.run(tf.nn.sigmoid(R, name="last")))

 print(sess.run(tf.nn.softmax(R,dim=-1, name="last")))

 print(sess.run(tf.nn.softplus(R,name="last")))

 print(sess.run(tf.nn.relu(R,name="XXX")))

 print(sess.run(tf.cast(T, tf.int32)))

 print(sess.run(tf.cast(T, tf.float32)))

 print(sess.run(tf.reduce_mean(tf.cast(T, tf.float32))))

==Tensorflow的reshape函数==

 tf.reshape()

 Y=tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32)

 sess.run(tf.global_variables_initializer())

 print(sess.run(Y))
[[1 2 3]
[4 5 6]]

 print(sess.run(tf.reshape(Y,[6])))
[ 1.  2.  3.  4.  5.  6.]


 print(sess.run(tf.reshape(Y,[3,2])))

<nowiki>array([[1 2]
       [3 4]
       [5 6]], dtype=float32)
</nowiki>
 print(sess.run(tf.reshape(Y,[-1])))

array([1 2 3 4 5 6],dtype=float32)

 print(sess.run(tf.reshape(Y, [-1,1])))

<nowiki>
array([[ 1.],
       [ 2.],
       [ 3.],
       [ 4.],
       [ 5.],
       [ 6.]], dtype=float32)
</nowiki>

==tf.argmax==

tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis.

 print(sess.run(tf.argmax(Y,1)))

[2 2]

 print(sess.run(tf.argmax(Y,0)))

[1 1 1]

=TensorFlow 实现线性代数运算=

TensorFlow数学运算

(https://www.tensorflow.org/api_guides/python/math_ops )

TensorFlow专门设计了线性代数（ Linear Algebra）运算加速器 XLA (Accelerated Linear Algebra)

*矩阵和向量的表示 matrix(matrices) and vector(vectors)
 （https://www.tensorflow.org/api_docs/python/tf/add ）
 （https://www.tensorflow.org/api_docs/python/tf/subtract ）

*矩阵加法和乘法 matrix addition and scalar multiplication

*矩阵和向量乘 matrix and vector multiplication
矩阵和矩阵乘 matrix and matrix multiplication

 (https://www.tensorflow.org/api_docs/python/tf/matmul )

*矩阵取逆矩阵和矩阵转置运算 matrix inverse/transpose

 (https://www.tensorflow.org/api_docs/python/tf/matrix_inverse )

=TensorFlow 模型保存与复原= 

TensorFlow Saver类用于保存与恢复模型参数。

<code>
# Add ops to save and restore all the variables.
 saver = tf.train.Saver()

  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)

  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
</code>


=TensorFlow 网站的教程（链接）=

==线性模型==

*TensorFlow 线性模型（Linear Model Tutorial）（Census数据集）

(https://www.tensorflow.org/tutorials/wide ) 

==分类模型==

*TensorFlow逻辑斯提回归（logistics regression model）（Census数据集）

(https://www.tensorflow.org/tutorials/wide#defining_the_logistic_regression_model )

==核方法==

*TensorFlow 显式核方法（Explicit Kernel Methods）

（https://www.tensorflow.org/tutorials/kernel_methods ）

==MLP/DNN/FCN ==

深度神经网络，多层感知机和全连接网络

*TensorFlow DNN分类器（Iris数据集）

（https://www.tensorflow.org/get_started/tflearn ）

==卷积网络==
*TensorFlow CNN分类器（MINIST数据集）

TF Layers （https://www.tensorflow.org/tutorials/layers ）

*TensorFlow深度卷积网络（CIFAR数据集）

（https://www.tensorflow.org/tutorials/deep_cnn ）

==循环网络==

*TensorFlow 语言模型建模（Language modelling）

https://www.tensorflow.org/tutorials/recurrent

*TensorFlow 序列对序列模型（seq2seq）

https://www.tensorflow.org/tutorials/seq2seq
