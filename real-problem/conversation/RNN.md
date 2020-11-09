RNN原理及简单实现

[toc]

# 1.基本的RNN（Recurrent Neural Network）

## 图示：

![RNNsimple](/Users/zhangxin/Desktop/Github/Deep-learning-example/img/RNNsimple.jpg)

## 计算方法：

$$
o_t=g(Vs_{t-1}) \\
s_t=f(Ux_t+Ws_{t-1})
$$

## 计算实例：
$$
o_t=g(Vs_{t}) \\
=Vf(Ux_t+Ws_{t-1}) \\
=Vf(Ux_t+WF(Ux_{t-1}+Ws_{t-2}))
$$


# 2.双向循环网络

## 图示：

![BRNN](/Users/zhangxin/Desktop/Github/Deep-learning-example/img/BRNN.png)

## 计算方法：

$$
y_2=g(VA_2+\dot{V}\dot{A_2}) \\
A_2=f(WA_1+Ux_2) \\
\dot{A_2}=f(\dot{W}\dot{A_3}+\dot{U}x_2)
$$

## 计算实例：


$$
o_t=g(Vs_t+\dot{V}\dot{s_t}) \\
s_t=f(Ux_t+Ws_{t-1}) \\
\dot{s_{t}}=f(\dot{U}x_t+\dot{W}\dot{s_{t+1}})
$$






