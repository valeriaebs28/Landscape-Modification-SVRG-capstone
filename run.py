import argparse
import os

parser = argparse.ArgumentParser(description="Run experiments and reproduce results.")
parser.add_argument('--SGD_lr_search', action='store_true')
parser.add_argument('--SVRG_lr_search', action='store_true')
parser.add_argument('--SVRGLM_lr_search', action='store_true')
parser.add_argument('--SVRGLMD_lr_search', action='store_true')
parser.add_argument('--SGD_small_batch_lr_search', action='store_true')
parser.add_argument('--SVRG_small_batch_lr_search', action='store_true')
parser.add_argument('--SVRGLM_small_batch_lr_search', action='store_true')
parser.add_argument('--SVRGLMD_small_batch_lr_search', action='store_true')
parser.add_argument('--SGD_tiny_batch_lr_search', action='store_true')
parser.add_argument('--SVRG_tiny_batch_lr_search', action='store_true')
parser.add_argument('--SVRGLM_tiny_batch_lr_search', action='store_true')
parser.add_argument('--SVRGLMD_tiny_batch_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SGD_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SVRG_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SVRGLM_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SVRGLMD_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SGD_small_batch_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SVRG_small_batch_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SVRGLM_small_batch_lr_search', action='store_true')
parser.add_argument('--CIFAR10_SVRGLMD_small_batch_lr_search', action='store_true')

LR_RANGE = [0.1, 0.03, 0.01, 0.003, 0.001]

def SGD_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SGD_lr_search',
        '--optimizer', 'SGD', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRG_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SVRG_lr_search',
        '--optimizer', 'SVRG', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRGLM_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py',
        '--exp_name', 'SVRGLM_lr_search',
        '--optimizer', 'SVRGLM', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRGLMD_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', #SEE IF CHANGE TO run_svrglm
        '--exp_name', 'SVRGLMD_lr_search',
        '--optimizer', 'SVRGLMD', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SGD_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SGD_small_batch_lr_search',
        '--optimizer', 'SGD', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRG_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SVRG_small_batch_lr_search',
        '--optimizer', 'SVRG', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRGLM_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SVRGLM_small_batch_lr_search',
        '--optimizer', 'SVRGLM', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRGLMD_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SVRGLMD_small_batch_lr_search',
        '--optimizer', 'SVRGLMD', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SGD_tiny_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SGD_tiny_batch_lr_search',
        '--optimizer', 'SGD', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '4',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRG_tiny_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SVRG_tiny_batch_lr_search',
        '--optimizer', 'SVRG', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '4',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRGLM_tiny_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SVRGLM_tiny_batch_lr_search',
        '--optimizer', 'SVRGLM', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '4',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def SVRGLMD_tiny_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'SVRGLMD_tiny_batch_lr_search',
        '--optimizer', 'SVRGLMD', 
        '--nn_model', 'MNIST_one_layer',
        '--dataset', 'MNIST',
        '--n_epoch', '100',
        '--batch_size', '4',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SGD_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SGD_lr_search',
        '--optimizer', 'SGD', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SVRG_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SVRG_lr_search',
        '--optimizer', 'SVRG', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SVRGLM_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SVRGLM_lr_search',
        '--optimizer', 'SVRGLM', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SVRGLMD_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SVRGLMD_lr_search',
        '--optimizer', 'SVRGLMD', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '128',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SGD_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SGD_small_batch_lr_search',
        '--optimizer', 'SGD', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SVRG_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SVRG_small_batch_lr_search',
        '--optimizer', 'SVRG', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SVRGLM_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SVRGLM_small_batch_lr_search',
        '--optimizer', 'SVRGLM', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

def CIFAR10_SVRGLMD_small_batch_lr_search():
    arg_list_template = [
        'python', 'run_svrglm.py', 
        '--exp_name', 'CIFAR10_SVRGLMD_small_batch_lr_search',
        '--optimizer', 'SVRGLMD', 
        '--nn_model', 'CIFAR10_convnet',
        '--dataset', 'CIFAR10',
        '--n_epoch', '100',
        '--batch_size', '16',
        '--weight_decay', '0.0001', 
        '--lr']
    for lr in LR_RANGE:
        arg_list = arg_list_template + [str(lr)]
        command = ' '.join(arg_list)
        print(command)
        os.system(command)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.SGD_lr_search:
        SGD_lr_search()
    
    if args.SVRG_lr_search:
        SVRG_lr_search()

    if args.SVRGLM_lr_search:
        SVRGLM_lr_search()

    if args.SVRGLMD_lr_search:
        SVRGLMD_lr_search()
    
    if args.SGD_small_batch_lr_search:
        SGD_small_batch_lr_search()
    
    if args.SVRG_small_batch_lr_search:
        SVRG_small_batch_lr_search()

    if args.SVRGLM_small_batch_lr_search:
        SVRGLM_small_batch_lr_search()

    if args.SVRGLMD_small_batch_lr_search:
        SVRGLMD_small_batch_lr_search()
    
    if args.SGD_tiny_batch_lr_search:
        SGD_tiny_batch_lr_search()
    
    if args.SVRG_tiny_batch_lr_search:
        SVRG_tiny_batch_lr_search()

    if args.SVRGLM_tiny_batch_lr_search:
        SVRGLM_tiny_batch_lr_search()

    if args.SVRGLMD_tiny_batch_lr_search:
        SVRGLMD_tiny_batch_lr_search()

    # CIFAR10
    if args.CIFAR10_SGD_lr_search:
        CIFAR10_SGD_lr_search()
    
    if args.CIFAR10_SVRG_lr_search:
        CIFAR10_SVRG_lr_search()

    if args.CIFAR10_SVRGLM_lr_search:
        CIFAR10_SVRGLM_lr_search()

    if args.CIFAR10_SVRGLMD_lr_search:
        CIFAR10_SVRGLMD_lr_search()
    
    if args.CIFAR10_SGD_small_batch_lr_search:
        CIFAR10_SGD_small_batch_lr_search()
    
    if args.CIFAR10_SVRG_small_batch_lr_search:
        CIFAR10_SVRG_small_batch_lr_search()

    if args.CIFAR10_SVRGLM_small_batch_lr_search:
        CIFAR10_SVRGLM_small_batch_lr_search()

    if args.CIFAR10_SVRGLMD_small_batch_lr_search:
        CIFAR10_SVRGLMD_small_batch_lr_search()