# Enhanced LSTM for Natural Language Inference
Source code for "Enhanced LSTM for Natural Language Inference" runnable on GPU and CPU based on Theano.
If you use this code as part of any published research, please acknowledge the following paper.

**"Enhanced LSTM for Natural Language Inference"**
Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. _ACL (2017)_ 

```
@InProceedings{Chen-Qian:2017:ACL,
  author    = {Chen, Qian and Zhu, Xiaodan and Ling, Zhenhua and Wei, Si and Jiang, Hui and Inkpen, Diana},
  title     = {Enhanced LSTM for Natural Language Inference},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver},
  publisher = {ACL}
}
```
Homepage of the Qian Chen, http://home.ustc.edu.cn/~cq1231/

The code is modified from [GitHub - nyu-dl/dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial/).

The code for tree-LSTM version has been released. Tree-LSTM part is modified from [GitHub - dallascard/TreeLSTM](https://github.com/dallascard/TreeLSTM), but support minibatches.

## Dependencies
To run it perfectly, you will need:
* Python 2.7
* Theano 0.8.2

## Running the Script
1. Download and preprocess 
```
cd data
bash fetch_and_preprocess.sh
```

2. Train and test model for ESIM
```
cd scripts/ESIM/
bash train.sh
```
3. Train and test model for TreeLSTM-IM
```
cd scripts/TreeLSTM-IM/
bash train.sh
```

The result is in `log.txt` file.
