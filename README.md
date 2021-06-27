# MNIST_Anomaly_Detection
Anomaly detection on MNIST dataset.

---
~~这个项目基于tensorflow2。要运行本程序，用conda install tensorflow或者conda install tensorflow-gpu安装相应的程序包。~~

classification.ipynb
最原始的MNIST数据集分类解决方法。

~~imbalanced_data.ipynb~~
~~解决不均衡样本分类的tensorflow方法。确保看懂每一步的思想与操作流程。最后的实现将MNIST数据集重新组织并喂到这个项目里即可，中间使用的模型可以参考classification.ipynb的模型来修改。~~

----
~~Updated 2021/6/13~~
~~可视化函数在`helper.py`中，用于作图。训练参数在`mnist.py`中，可修改设置异常值的数字`anomaly_num`。~~

~~不知道什么原因，每次调用完`main.py`下面的`baseline()`之后`matplotlib`的作图环境会发生改变，以至于画`class_weights()`的训练过程的图很难看，因此在`main.py`当中分两次跑，每次注释掉另外一个训练流程，得到最后结果。~~

~~训练结果保存在两个txt文件中。训练过程以及混淆矩阵已经用图像可视化保存好。~~

----
### Updated 2021/6/27

- 重构了网络模型的建模代码、训练代码。
- 修复了绘图出错、不能一次性进行三种方法训练的问题。
- 将预测数据保存到了`./result`文件目录下。

#### Environment
- Ubuntu 18.04 LTS
- NVIDIA GTX 1060, CUDA 11.1

#### Requirements
- **Tensorflow 2.x**

    Run `conda install tensorflow-gpu`

- **Other requirements**

    Run `pip install -r requirements`
    
#### Get started
Run `python train.py`, predictions would be saved under `./result` directory.

### Details
- **model.py**

    CNN network built here.

- **plot.py**

    Prediction visulizations.

- **method.py**

    Three different training method applied.

- **train.py**

    Data loading and main train stages.

- **vis.py**

    About MNIST: anomaly data distribution visualizations.