
##### 표준편차가 0.1이 되도록 shape의 형상으로 랜덤한 값을 넣어준다. Outputs random values from a truncated normal distribution. The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
```python
initial = tf.truncated_normal(shape, stddev=0.1)
```
##### 아래와 같이 하면 값을 얻을 수 있다.
```python
    initial = tf.truncated_normal([2,2],mean=0, stddev=0.1)
    sess = tf.Session()
    result=sess.run(initial)
    print(result)
    print(sum(result))
```

##### GPU로 실행했을 때 아래 코드를 실행하면 다음과 같은 에러가 뜬다.
```python
 def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```
```
ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[10000,32,28,28] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[Node: Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](Reshape, Variable/read)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
```

##### GPU의 메모리가 10000개나 되는 test data를 한번에 처리하기 어렵기 때문에 이런 에러가 뜬 것 같다

##### 텐서보드 사용법 - 만약 로그의 위치가 D:\java-neon\eclipse\python\CNNUsingTensorFlow\CNN\board\sample 안에 있다면
##### cmd에 D:\java-neon\eclipse\python\CNNUsingTensorFlow\CNN\board>tensorboard --logdir=sample/
##### 라고 치면 된다.

##### ConvNet ver 1과 ver 2가 차이가 난다. ver 1은 잘 돌아가는데 ver 2는 학습이 전혀 안된다. tensorflow를 제대로 사용하지 못해서일까?
##### 그 이유는 바로 learning rate 때문이었음, ver 1에서는 0.0001로 정해주고 ver 2에서는 0.1로 정하였는데 깊이가 깊은 신경망에서는 learning_rate을 낮게 정해주는 것이 좋은 것 같음, 그렇지 않으면 delta w가 발산하게 됨 (w = w - a*delta_w)
##### hyperparameter의 중요성을 뼈저리게 깨달음

##### ConvNet ver 1을 다음과 같은 조건으로 돌려보았다.
##### Conv - Pool - Conv - Pool - fc - fc - softmax
##### mini_batch = 100, epoch = 2000, GradientDescentdent, learning_rate = 0.0001 -> test data accuracy : 0.7396
##### mini_batch = 100, epoch = 2000, GradientDescentdent, learning_rate = 0.0001, Dropout -> test data accuracy : 0.6763
##### mini_batch = 100, epoch = 2000, Adam, learning_rate = 0.0001 -> test data accuracy : 0.9812
##### mini_batch = 100, epoch = 2000, Adam, learning_rate = 0.0001, Dropout -> test data accuracy : 0.9822

##### ConvNet ver 2를 다음과 같은 조건으로 돌려보았다.
##### mini_batch = 100, epoch = 2000, GradientDescentdent, learning_rate = 0.0001 -> test data accuracy : 0.8754
##### mini_batch = 100, epoch = 2000, GradientDescentdent, learning_rate = 0.0001, Dropout -> test data accuracy : 0.8736
##### mini_batch = 100, epoch = 2000, GradientDescentdent, learning_rate = 0.0001, batch_normalization -> test data accuracy : 0.892
##### mini_batch = 100, epoch = 2000, Adam, learning_rate = 0.0001 -> test data accuracy : 0.9863
##### mini_batch = 100, epoch = 2000, Adam, learning_rate = 0.0001, Dropout -> test data accuracy : 0.9878
##### mini_batch = 100, epoch = 2000, Adam, learning_rate = 0.0001, batch_normalization -> test data accuracy : 0.9878
