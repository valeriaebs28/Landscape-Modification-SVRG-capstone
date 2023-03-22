import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

experiments = ["CIFAR10_SVRG_lr_search", "CIFAR10_SVRGLMD_lr_search", "CIFAR10_SVRGLM_lr_search",
               "CIFAR10_SVRG_small_batch_lr_search", "CIFAR10_SVRGLMD_small_batch_lr_search", "CIFAR10_SVRGLM_small_batch_lr_search"]

data_dir = "outputs"
# load data
all_stats = []
for exp in experiments:
    folders = [x for x in os.listdir(data_dir) if x.startswith(exp)]
    
    for folder in folders:
        #print(folder)
        stats = {}
        args_file = os.path.join(data_dir, folder, "args.json")
        with open(args_file, "r") as f:
            args = json.load(f)
        #print(args)
        npz_file = os.path.join(data_dir, folder, "train_stats.npz")
        npz = np.load(npz_file)
        stats['epoch'] = pd.Series(np.arange(len(npz['train_loss'])))
        stats['train_loss'] = pd.Series(npz['train_loss'])
        stats['train_acc'] = pd.Series(npz['train_acc'])
        stats['val_loss'] = pd.Series(npz['val_loss'])
        stats['val_acc'] = pd.Series(npz['val_acc'])
        stats = pd.DataFrame(stats)

        op_function =''

        # use for main comparison; use when comparing f1 vs f2, SVRGLMD SVRGLM
        # if folder.endswith('f1'):
        #     op_function = '-f1'
        # elif folder.endswith('f2'):
        #     op_function = '-f2'

        # use for extra study on c, used f1
        # if folder.endswith('_c05_f1'):
        #     op_function = ', c=0.5'
        # elif folder.endswith('_c025_f1'):
        #     op_function = ', c=0.25'
        # elif folder.endswith('_c075_f1'):
        #     op_function = ', c=0.75'
        # elif folder.endswith('_c2_f1'):
        #     op_function = ', c=2'
        # elif folder.endswith('_ad2_f1'):
        #     op_function = '_ad2'
        # elif folder.endswith('_c01_f1'):
        #     op_function = ', c=0.1'


        stats['optimizer'] = args['optimizer'] + op_function

        stats['learning_rate'] = args['lr']
        stats['batch_size'] = args['batch_size']
        all_stats.append(stats)
stats_df = pd.concat(all_stats)
stats_df.head()

stats_df = stats_df[stats_df['learning_rate'] <= 0.1]


# plot results
sns.set(palette="muted", font_scale=1.1)
sns.set_style("ticks")

# # training loss
# # g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=4.9, aspect=1.25)
g = sns.FacetGrid(stats_df, col="learning_rate", row='batch_size', hue='optimizer',height=8, aspect=1)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    #plt.ylim(0, 0.5)
g = g.map(plot_loss, "epoch", "train_loss").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Training Loss (CIFAR10)", fontsize=15)
g.savefig("figures/cifar10_lr_search_train_loss.png")

# # training accuracy
# # g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=4.9, aspect=1.25)
g = sns.FacetGrid(stats_df, col="learning_rate", row='batch_size', hue='optimizer',height=8, aspect=1)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    plt.ylim(0.5, 1)
g = g.map(plot_loss, "epoch", "train_acc").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Training Accuracy (CIFAR10)", fontsize=15)
g.savefig("figures/cifar10_lr_search_train_acc.png")

# # validation accuracy
# # g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=4.9, aspect=1.25)
g = sns.FacetGrid(stats_df, col="learning_rate", row='batch_size', hue='optimizer',height=8, aspect=1)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    plt.ylim(0.4, 0.7)
g = g.map(plot_loss, "epoch", "val_acc").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Validation Accuracy (CIFAR10)", fontsize=15)
g.savefig("figures/cifar10_lr_search_val_acc.png")




# used to get the csv files for information for tables

# df_copy = stats_df.copy()
# # last_epoch = max(df_copy['epoch'])
# # df_copy = df_copy[df_copy['epoch']== last_epoch]
# # print("we print EXTRA's epoch 100: ")
# # print(df_copy.to_string())

# # os.makedirs('folder/subfolder', exist_ok=True)  
# df_copy.to_csv('csv_files/main_experiments_CIFAR10_extended_epochs.csv')  