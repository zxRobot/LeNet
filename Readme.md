## 全连接层的反向传播
下面介绍一下全连接层的反向传播算法的推导。首先对全连接层的前向过程进行一下介绍。

![avatar](pic/fc_f.png )

我们把全连接层的每一层神经元都表示为一个列向量。上一层的神经元的输出，通过乘上当前层的权重矩阵加上列向量形式的偏置项，得到激活前的输出值，最后通过激活函数得到当前层的输出，公式如下：
$$z^l=W^la^{l-1}+b^l$$
$$a^l=\sigma (z^l)$$
其中 $z^l$表示第$l$层未经过激活函数的结果，$a^l$表示经过激活函数得到的输出结果。假设上一层的输出是[m,1]的列向量，当前层的输出是[n,1]的列向量，那权重矩阵应为[n,m],偏置矩阵为[n,1]。
接下来进行反向传播的过程，首先定义一个误差函数，来衡量神经网络与正确的输出之间的差异。为了简单，本文直接采用了最简单的均方误差损失函数：
$$C=\frac{1}{2}||a^L-y||^2$$ 
其中$L$代表多层感知机的总层数，得到总误差之后，我们就可以通过反向传播对各层的权重矩阵$W^l$和偏置列向量$b^l$进行更新，使神经网络的误差减小，达到训练的目的。  
由于反向传播链式传导的规律，为了避免重复计算，我们引入中间量$\delta^l$,我们称它为第$l$层的误差，具体含义为误差函数对于神经网络第$l$层未经激活函数的输出值的偏导数，即$\delta^l=\frac{\partial C}{\partial z^l}$，输出层的网络误差$\delta ^L$ 为：
$$\delta ^L=\frac{\partial C}{\partial z^L}=\frac{\partial C}{\partial a^L}\frac{\partial a^L}{\partial z^L}=(a^L-y)\bigodot \sigma^l(z^l)$$
其中，$\bigodot$表示矩阵逐元素相乘。  
接下来求$W$矩阵的导数，应用链式法则，得：
$$\frac{\partial C}{\partial W^l}=\frac{\partial C}{\partial z^l}\frac{\partial z^l}{\partial W^l}=\delta^L(a^{L-1})^T$$
$$\frac{\partial C}{\partial b^l}=\frac{\partial C}{\partial z^l}\frac{\partial z^l}{\partial b^l}=\delta^L\bigodot1=\delta^L$$
矩阵乘法的求导即乘上系数矩阵所对应的转置，同时左乘还是右乘需要跟上述前向过程保持一致。
我们得到了最后一层的误差，接下来我们根据后一层的 $\delta$ 得到前一层的$\delta$，最后我们可以求得每一层的$\delta$。假设我们得到了第 $l+1$ 的 $\delta$ ，第$l$层的 $\delta$ 表示如下：
$$\delta^l=\frac{\partial C}{\partial z^l}=\frac{\partial C}{\partial z^{l+1}}\frac{\partial z^{l+1}}{\partial z^l}=\delta^{l+1}\frac{\partial z^{l+1}}{\partial z^l}$$ 
$$z^{l+1}=W^{l+1}a^l+b^{l+1}=W^{l+1}\delta(z^l)+b^{l+1}$$
$$\delta^l=(W^{l+1})^T\delta^{l+1}\bigodot\delta'(z^l)$$
这同时也要注意求导后矩阵运算是左乘还是右乘。
接下来我们分析误差函数$C$对每一层参数$W$的梯度
$$\frac{\partial C}{\partial w^l}=\frac{\partial C}{\partial z^l}\frac{\partial z^l}{\partial w^l}=\delta^l(a^{l-1})^T$$
$$\frac{\partial C}{\partial b^l}=\frac{\partial C}{\partial z^l}\frac{\partial z^l}{\partial b^l}=\delta^l$$
接下来通过梯度下降法更新权重和偏置
$$W^l=W^l-\eta\frac{\partial C}{\partial w^l}$$
$$b^l=b^l-\eta\frac{\partial C}{\partial b^l}$$
在上述的分析中，我们只根据一组训练数据更新权重，在一般情况下，我们往往会采用随即梯度下降法，一次性训练一批数据，先计算一批数据的中每一组数据的误差，在根据它们的平均值来进行权重更新
$$W^l=W^l-\frac{\eta}{batch_size}\sum{\frac{\partial C}{\partial W^l}}$$

```python
def full_connect(self,input_data,fc,front_delta=None,deriv=False):
    N=input_data.shape[0]
    if deriv==False:
        output_data=np.dot(input_data.reshape(N,-1),fc)
        return output_data
    else:
        back_delta=np.dot(front_delta,fc.T).reshape(input_data.shape)
        fc+=self.lr*np.dot(input_data.reshape(N,-1),front_delta)
        return back_delta,fc
```
## 卷积层的反向传播
①卷积层通过张量的卷积，或者说是多个矩阵卷积求和得到的输出，这和全连接层是不同的,所以在反向传播的时候，上一层的$\delta^{l-1}$递推计算方法是不同的。  
②在卷积运算的过程中，从$\delta^l$推导出$W$、$b$的方式也是不同的。  
各个符号所代表的数学意义和上一节全连接层相同。  
卷积层的前向传播过程如下：  

<img src="https://i2.wp.com/img-blog.csdnimg.cn/20200509131408933.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE4NzI0ODQx,size_16,color_FFFFFF,t_70" div align=center />  

前向传播的公式为：  
$$a^l=\sigma(z^l)=\sigma(a^{l-1}*W^l+b^l)$$
在全连接中，$\sigma^l$和$\sigma^{l+1}$的关系为：
$$\delta^l=\frac{\partial C}{\partial z^l}=\frac{\partial C}{\partial z^{l+1}}\frac{\partial z^{l+1}}{\partial z^l}=\delta^{l+1}\frac{\partial z^{l+1}}{\partial z^l}$$ 
同样应用在卷积层中，但如上面①提到的，$\frac{\sigma z^{l+1}}{\sigma z^l}$在全连接层和卷积层的计算方法不同。我们通过一个简单的例子来进行一下分析。  
假设我们$l$层的输入$a^{l}$是一个$3\times3$的矩阵，第$l+1$层的卷积核$W^l$是一个$2\times2$的矩阵，步长为1，则输出$z^{l+1}$为：
$$z^{l+1}=a^l*W^{l+1}$$ 
$\begin{pmatrix}
    a_{11}&a_{12}&a_{13}\\
    a_{21}&a_{22}&a_{23}\\
    a_{31}&a_{32}&a_{33}\\
\end{pmatrix}$*$\begin{pmatrix}
    w_{11}&w_{12}\\
    w_{21}&w_{22}\\
\end{pmatrix}$=$\begin{pmatrix}
    z_{11}&z_{12}\\
    z_{21}&z_{22}\\
\end{pmatrix}$ 
  
根据卷积的计算公式，得
$$z_{11}=a_{11}*w_{11}+a_{12}*w_{12}+a_{21}*w_{21}+a_{22}*w_{22}$$
$$z_{12}=a_{12}*w_{11}+a_{13}*w_{12}+a_{22}*w_{21}+a_{23}*w_{22}$$
$$z_{21}=a_{21}*w_{11}+a_{22}*w_{12}+a_{31}*w_{21}+a_{32}*w_{22}$$
$$z_{22}=a_{22}*w_{11}+a_{23}*w_{12}+a_{32}*w_{21}+a_{33}*w_{22}$$
模拟反向求导，得
$$\nabla a^l=\frac{\partial C}{\partial a^l}=\frac{\partial C}  {\partial z^{l+1}}  \frac{\partial z^{l+1}}{\partial a^l}=\delta^{l+1} \frac{\partial z^{l+1}}{\partial a^l}$$
从上式可以看出，对于$a^l$的梯度误差$\nabla a^{l}$，等于$\delta^{l+1}  \frac{\partial z^{l+1}}{\partial a^l}$,而$\frac{\partial z^{l+1}}{\partial a^l}$可以通过上述的卷积计算式推到出来。我们输出了一个$2\times2$的矩阵，所以传播误差也为一个$2\times2$的矩阵，假设为
$$
\begin{pmatrix}
\delta_{11}&\delta_{12}\\
\delta_{21}&\delta_{22}\\
\end{pmatrix}
$$
对于$a_{11}$的梯度，在上述算式中只有$z_{11}$和他有关系。$\delta_{11}$实际上和$z_{11}$所代表的通道是同一通道。
$$\nabla a_{11}=\delta_{11}w_{11}$$
$$\nabla a_{12}=\delta_{11}w_{12}+\delta_{12}w_{11}$$
$$\nabla a_{13}=\delta_{12}w_{12}$$
$$\nabla a_{21}=\delta_{11}w_{21}+\delta_{21}w_{11}$$
$$\nabla a_{22}=\delta_{11}w_{22}+\delta_{12}w_{21}+\delta_{21}w_{12}+\delta_{22}w_{11}$$
$$\nabla a_{23}=\delta_{12}w_{22}+\delta_{22}w_{12}$$
$$\nabla a_{31}=\delta_{21}w_{21}$$
$$\nabla a_{32}=\delta_{21}w_{22}+\delta_{22}w_{21}$$
$$\nabla a_{33}=\delta_{22}w_{22}$$

表示为矩阵协相关的形式表示（其实卷积层的卷积实际上是数学的协相关）：  


$\begin{pmatrix}
    \nabla a_{11}&\nabla a_{12}&\nabla a_{13}\\
    \nabla a_{21}&\nabla a_{22}&\nabla a_{23}\\
    \nabla a_{31}&\nabla a_{32}&\nabla a_{33}\end{pmatrix}$=$\begin{pmatrix}
    0&0&0&0\\
    0&\delta_{11}&\delta_{12}&0\\
    0&\delta_{21}&\delta_{22}&0\\
    0&0&0&0\end{pmatrix}$*$\begin{pmatrix}
    w_{22}&w_{21}\\
    w_{12}&w_{11}
\end{pmatrix}$  
为了符合梯度计算，我们在误差矩阵周围填充了一圈0，此时我们将卷积核反转180度之后和本层的梯度误差进行卷积，就可以得到下一层的梯度误差。
$$\delta^l=\frac{\partial C}{\partial z^l}=\frac{\partial C}{\partial z^{l+1}}\frac{\partial z^{l+1}}{\partial z^l}=\delta^{l+1}\frac{\partial z^{l+1}}{\partial z^l}=\delta^{l+1}*rot180(W^{l+1})$$ 

现在我们推导完了误差的反向传播关系，现在我们根据梯度误差来对$W$、$b$进行更新。  

$$z^{l+1}=a^{l}*W^{l+1}+b$$
$$\frac{\partial C}{\partial W^{l+1}}=a^l*\delta ^{l+1}$$  

但是卷积层输入的是矩阵，还是根据上述那个例子来进行分析，可得：  

$$\frac{\partial C}{\partial W^{l+1}_{11}}=a_{11}\delta_{11}+a_{12}\delta_{12}+a_{21}\delta_{21}+a_{22}\delta_{22} $$
$$\frac{\partial C}{\partial W^{l+1}_{12}}=a_{12}\delta_{11}+a_{13}\delta_{12}+a_{22}\delta_{21}+a_{23}\delta_{22} $$
$$\frac{\partial C}{\partial W^{l+1}_{21}}=a_{21}\delta_{11}+a_{22}\delta_{12}+a_{31}\delta_{21}+a_{33}\delta_{22} $$
$$\frac{\partial C}{\partial W^{l+1}_{12}}=a_{22}\delta_{11}+a_{23}\delta_{12}+a_{32}\delta_{21}+a_{33}\delta_{22} $$

$\frac{\partial C}{\partial W^{l+1}}=$ $\begin{pmatrix}
    a_{11}&a_{12}&a_{13}\\
    a_{21}&a_{22}&a_{23}\\
    a_{31}&a_{32}&a_{33}\\
\end{pmatrix}$*$\begin{pmatrix}
    \delta_{11}&\delta_{12}\\
    \delta_{21}&\delta_{22}\\
\end{pmatrix}$  

对于$b$，因为$\delta^{l+1}$是高维张量，$b$是一个向量，在这将$\delta^{l+1}$的各个子矩阵的项相加，得到一个误差向量，即为$b$的梯度。
```python
def convolution(self,input_data,kernel,front_delta=None,deriv=False):
    N,C,W,H=input_data.shape
    K_NUM,K_C,K_W,K_H=kernel.shape
    if(deriv==False):
        output_data=np.zeros((N,K_NUM,W-K_W+1,H-K_H+1))
        for imgID in range(N):
            for Kid in range(K_C):
                for Cid im range(C):
                    output_data+=convolve2d(input_data[imgID][Cid],kernel[Kid][Cid],mode='valid')
        return output_data
    else:
        back_delta=np.zeros((N,C,W,H))
        kernel_gradient=np.zeros((K_NUM,K_C,K_W,K_H))
        padded_front_delta=np.pad(front_delta,[(0,0), (0,0), (K_W-1, K_H-1), (K_W-1, K_H-1)], mode='constant', constant_values=0)
        for imgId in range(N):
            for cId in range(C):
                for kId in range(K_NUM):
                    back_delta[imgId][cId] += convolve2d(padded_front_delta[imgId][kId], kernal[kId,cId,::-1,::-1], mode='valid')
                    kernal_gradient[kId][cId] += convolve2d(front_delta[imgId][kId], input_map[imgId,cId], mode='valid')
            # update weights
        kernal += self.lr * kernal_gradient
        return back_delta, kernal

```