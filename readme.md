#  DCCA, DCCAE
This repo contains a pytorch implementation of DCCA and DCCAE.

### dataset
You can download the dataset from [noisymnist_view1.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz) and [noisymnist_view2.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz), or use the download_data.sh.Put them in dataset/..

### model
You can download from https://pan.baidu.com/s/1l_SM96jLSu1cpuYXYlzPuA?pwd=kxr5 
提取码：kxr5 ,and put them in Dcca/DCCA_checkpoint.model and Dccae/DCCAE_checkpoint.model respectively.

### Run
DCCA:

    train: run Dcca/DCCA_train.py
    test:  run Dcca/DCCA_test.py

DCCAE:

    train: run Dccae/DCCAE_train.py
    test:  run Dccae/DCCAE_test.py

### References:

* [DeepCCA](https://github.com/Michaelvll/DeepCCA)
* [On Deep Multi-View Representation Learning](http://proceedings.mlr.press/v37/wangb15.pdf)
