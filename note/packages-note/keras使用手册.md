keras使用手册

一个十分好的参考资料： https://github.com/czy36mengfei/tensorflow2_tutorials_chinese

[toc]

# 导入tensorflow2.0以及keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)
```

# 1.模型的序列化API

## 参考链接：

https://blog.csdn.net/zhong_ddbb/article/details/107658686

## 用途：

序列化模型常用于：简单的神经网络层的堆叠。每一层都有一个输入tensor和一个输出tensor。

## 函数定义：

```python
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        keras.Input(shape=(3,3)),
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
```

## 单独定义层后堆积：

```python
# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
```

## 一层层添加：

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

## 模型细节查看

### 查看形状：

```python
model.summary()  # 在模型构建过程中随时查看output Shape
```

### 查看权重：

```python
model.weights
```

## 序列化API型缺陷

序列化模型不适用以下情况：

(1) 模型含有多个输入或多个输出

(2) 模型中的某一层含有多个输入或多个输出

(3) 模型中存在层共享的情况

(4) 模型中存在类似**残差网络**这样的非线性模型。

# 2.模型的函数式API

函数式API可以处理具有非线性拓扑的模型（残差神经网络），具有共享层的模型以及具有多个输入或输出的模型。

## 参考链接：

https://blog.csdn.net/zhong_ddbb/article/details/107658686

## 基本步骤：

### 创建一个输入节点，常见的输入有以下两种：

```python
# 输入是一个784维的向量，样本数不用指定，
inputs = keras.Input(shape=(784,))
# 输入是一个大小为32x32的3通道图片
img_inputs = keras.Input(shape=(32, 32, 3))
```

可以通过：inputs.shape，inputs.dtype来查看输入节点基本细节。

### 定义其他网络层，以函数调用的形式同时传入inputs:

```python
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
```

### 指定输入输出，创建模型:

```python
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

## 例子：

### 1.图片的编码解码器

先定义一个encoder：

```python
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()
```

再定义一个解码器decoder：

```python
decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()
```

将图片先输入编码器，对输出结果进行解码，形成新的模型autoencoder

```python
autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
```

### 2.以下是将一组模型合并为一个平均其预测的模型的方法：

```python
def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
```

### 3.层共享:

共享层通常用于编码来自相似空间的输入（例如，两个具有相似词汇的不同文本）。它们可以在这些不同的输入之间共享信息，并且可以在更少的数据上训练这种模型。下面是两个不同的文本输入共享的嵌入层：

```python
def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
```

# 3.过拟合处理办法:正则化以及Dropout

## 参考链接：

 [Keras Dropout和正则化的使用](https://www.cnblogs.com/XUEYEYU/p/keras-learning-4.html)

## 使用办法：

1.dropout

```python
from keras.layers import Dense,Dropout  #在这里导入dropout
model=Sequential([
    Dense(units=200,input_dim=784,bias_initializer='zeros',activation='tanh'), #双曲正切激活函数
    #Dropout(0.4),  #百分之40的神经元不工作
    Dense(units=100,bias_initializer='zeros',activation='tanh'), #双曲正切激活函数
    #Dropout(0.4),  #百分之40的神经元不工作
    Dense(units=10,bias_initializer='zeros',activation='softmax') 
])
```

2.正则化：

```python
from keras.regularizers import l2  #导入正则化l2（小写L）
model=Sequential([
    #加上权值正则化
    Dense(units=200,input_dim=784,bias_initializer='zeros',activation='tanh',kernel_regularizer=l2(0.0003)), #双曲正切激活函数
    Dense(units=100,bias_initializer='zeros',activation='tanh',kernel_regularizer=l2(0.0003)), #双曲正切激活函数
    Dense(units=10,bias_initializer='zeros',activation='softmax',kernel_regularizer=l2(0.0003)) 
])
```

# 4.可视化显示：

## 参考链接：

https://keras.io/zh/visualization/

## 使用：

### 模型可视化:

将绘制一张模型图，并保存为文件：

```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model` 有 4 个可选参数:

- `show_shapes` (默认为 False) 控制是否在图中输出各层的尺寸。
- `show_layer_names` (默认为 True) 控制是否在图中显示每一层的名字。
- `expand_dim`（默认为 False）控制是否将嵌套模型扩展为图形中的聚类。
- `dpi`（默认为 96）控制图像 dpi。

### 训练历史可视化:

Keras `Model` 上的 `fit()` 方法返回一个 `History` 对象。`History.history` 属性是一个记录了连续迭代的训练/验证（如果存在）损失值和评估值的字典。这里是一个简单的使用 `matplotlib` 来生成训练/验证集的损失和准确率图表的例子：

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

## 报错及解决：

### 报错：

```
`pydot` failed to call GraphViz.Please install GraphViz (https://www.graphviz.org/) and ensure that its executables are in the $PATH.
```

### 解决：

首先下载下面三个模块

```
pip install graphviz

pip install pydot

pip install pydot_ng
```

下载这三个还不能解决这个问题，还需要安装GraphViz

直接输入命令：conda install GraphViz --channel conda-forge -y

# 5.模型的训练和评估

## 加载数据集

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
```

## 模型的compile：

### 示例：

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

或者：

```python
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
```

### 函数作用：

optimizer：目标函数，优化器，如Adam

loss：计算损失，这里用的是交叉熵损失

metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。如果要在多输出模型中为不同的输出指定不同的指标，可向该参数传递一个字典，例如metrics={‘output_a’: ‘accuracy’}

### 参考链接：

https://blog.csdn.net/WWWWWWGJ/article/details/86409329

https://blog.csdn.net/zhong_ddbb/article/details/107658686

https://blog.csdn.net/sinat_16388393/article/details/93207842

## 模型的训练：

### 示例：

```python
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)
```

可以通过history查看训练的损失值：

```python
history.history
```

当训练数据数Numpy时，也可以事先不做验证集切分，直接通过validation_split设置验证集的大小：

```python
model = get_compiled_model()
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
```

### 在训练过程中可以指定类别权重和样本权重

指定类别权重：

```python
import numpy as np

class_weight = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    # Set weight "2" for class "5",
    # making this class 2x more important
    5: 2.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}

print("Fit with class weight")
model = get_compiled_model()
model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=1)
1234567891011121314151617181920
```

指定样本权重：

```python
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

print("Fit with sample weight")
model = get_compiled_model()
model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=1)
```

### 参考链接：

https://blog.csdn.net/zhong_ddbb/article/details/107658686

## 模型的评估、预测:

```python
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
```





