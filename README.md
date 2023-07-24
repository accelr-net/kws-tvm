# KWS PyTorch Network Implemented in TVM
***

[TVM](https://tvm.apache.org/) is an open-source deep learning compiler stack that enables efficient deployment of deep learning models on various hardware platforms.

In this repository we present the necessary script files to the optimize inference run of a Keyword Spotting (KWS) ResNet-18 PyTorch network using [Apache TVM](https://tvm.apache.org/docs/tutorial/introduction.html#sphx-glr-tutorial-introduction-py)

The KWS PyTorch network implemented in this repository is designed to recognize specific keywords in audio data set. This particular KWS network creates a  mel-spectogram of the audio data as a preprocessing step before feeding it to the ResNet-18 network. There are a total of 35 output classes and these words are predicted with 89% accuracy in both the original PyTorch implementation as well as the target LLVM device implemented via TVM.

![Typical TVM Flow Diagram](https://raw.githubusercontent.com/apache/tvm-site/main/images/tutorial/overview.png "Typical TVM Flow Diagram")
*The above diagram is directly refers to a resource from the [Apache TVM](https://raw.githubusercontent.com/apache/tvm-site/main/images/tutorial/overview.png) github repo.*
***

## Prerequisites

### 1. Installation of TVM
There are 3 ways to install the TVM.

+ For a quick try out on TVM you can [install from Docker](https://tvm.apache.org/docs/install/docker.html#docker-images).
+ Or locally through [install from the binary package](https://tvm.apache.org/docs/tutorial/install.html#sphx-glr-tutorial-install-py).
+ Otherwise [install from source](https://tvm.apache.org/docs/install/from_source.html#install-from-source) with maximum flexibility 
 to configure the build from official source releases.

### 2. Installation of Pytorch Packages

These 3 packages need to be installed.
```
pip install torch torchvision torchaudio
```
### 3. Pre-trained Pytorch Network files.

Our scripts require a pre-trained network - i.e. a model file (`[model].pt`) and a label file (`[label].pickle`) to operate. These binary files are not provided with the repo but can be genearated through model training scripts provided in the training directory.
Once the model is trained, the `[model].pt` file must be saved as trace model after training because TVM does not support other saved models from pytorch.



***

## Process
+ Clone the repository.
+ Run the `training.py` script from `kws-tvm\kws\training` folder to genrate trace model and pickle file.
```
python training.py 
```
+ If the training script is running for the first time and to to download the dataset, set the download value to `TRUE` in the following line. 
```
super().__init__("./", download=True)
```
+ Run the `kwsrn18pytorch.py` script from `kws-tvm\kws\rn18` folder with all necessary files.
 ```
python kwsrn18pytorch.py
```
+ For better auto tuning optimization TVM Runner parameters can be changed accordingly.
+ Different data sets can be tested by renaming the testset literals.

***

## References
1. [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)
2. [Compiling Pytorch Models in TVM](https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html#sphx-glr-how-to-compile-models-from-pytorch-py)
