# SVRG-LM for neural networks (PyTorch)

This code was modified from [1] in order to implement different landscape modifications to Stochastic Variance Reduced Gradient (SVRG).

[1] Rie Johnson and Tong Zhang. “Accelerating stochastic gradient descent using predictive variance reduction”. In: Advances in neural information processing systems 26 (2013). URL: https://papers.nips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf.

### Train neural networks with SVRG
```
python run_svrglm.py --optimizer SVRGLM --nn_model CIFAR10_convnet --dataset CIFAR10 --lr 0.01
python run_svrglm.py --optimizer SVRGLMD --nn_model CIFAR10_convnet --dataset CIFAR10 --lr 0.01
python run_svrglm.py --optimizer SVRGLM --nn_model MNIST_one_layer --dataset MNIST --lr 0.01
python run_svrglm.py --optimizer SVRGLMD --nn_model MNIST_one_layer --dataset MNIST --lr 0.01
python run_svrglm.py --optimizer SVRG --nn_model MNIST_one_layer --dataset MNIST --lr 0.01
```

### Run experiments to compare SVRG vs. SVRGLM
```
python run.py --CIFAR10_SVRGLM_lr_search
python run.py --CIFAR10_SVRGLMD_lr_search
python run.py --CIFAR10_SVRG_lr_search
python run.py --SVRGLM_small_batch_lr_search
python run.py --SVRGLMD_small_batch_lr_search
python run.py --SVRG_small_batch_lr_search

```

### Plot the training curve results
```
python plot_mnist_results.py 
python plot_cifar_results.py 
```
