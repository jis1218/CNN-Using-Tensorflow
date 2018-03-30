
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