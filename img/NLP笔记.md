[toc]

# 1.数据预处理部分：

## 分词器分词：

使用分词器将长文本进行分词，是的长文本数据变为多个词汇组成的数据。

常用jieba分词器。

```python
import jieba
jieba.cut(string)
```

## 独热编码：

将标签进行独热编码，使得标签变为向量结果，将分类问题能够被神经网络进行使用。

常用sklearn库中的函数进行操作。

```python
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
data_y = le.fit_transform(answer_ext.score).reshape(-1,1)

ohe = OneHotEncoder()
data_y = ohe.fit_transform(data_y).toarray()
```

## 建立词典库：

使得词汇转化为数字，从而可以进一步的将文本数据转化为向量。

常用sklraen中的库操作。

```python
from keras.preprocessing.text import Tokenizer
max_words = 2000

tok = Tokenizer(num_words=max_words)  ## The max word number is 2000
tok.fit_on_texts(answer_ext.answer_cut.astype(str))
```

## 长文本转化为向量：

利用分词结果、词典库，将文本转化为向量的形式。

常用pandas中数据进行操作。

```python
max_len = 40
data_seq = tok.texts_to_sequences(answer_ext.answer_cut.astype(str))
data_seq_mat = sequence.pad_sequences(data_seq,maxlen=max_len)
```





















