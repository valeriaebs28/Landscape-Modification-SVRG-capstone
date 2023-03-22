import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

experiments = ["SVRG_lr_search", "SVRGLMD_lr_search", "SVRGLM_lr_search",
               "SVRG_small_batch_lr_search", "SVRGLMD_small_batch_lr_search","SVRGLM_small_batch_lr_search"]

data_dir = "outputs"

# load data
all_stats = []
for exp in experiments:
    folders = [x for x in os.listdir(data_dir) if x.startswith(exp)]
    
    for folder in folders:
        stats = {}
        args_file = os.path.join(data_dir, folder, "args.json")
        with open(args_file, "r") as f:
            args = json.load(f)

        npz_file = os.path.join(data_dir, folder, "train_stats.npz")
        npz = np.load(npz_file)
        stats['epoch'] = pd.Series(np.arange(len(npz['train_loss'])))
        stats['train_loss'] = pd.Series(npz['train_loss'])
        stats['train_acc'] = pd.Series(npz['train_acc'])
        stats['val_loss'] = pd.Series(npz['val_loss'])
        stats['val_acc'] = pd.Series(npz['val_acc'])
        stats = pd.DataFrame(stats)

        op_function =''
        if folder.endswith('f1'):
            op_function = '-f1'
        elif folder.endswith('f2'):
            op_function = '-f2'
        stats['optimizer'] = args['optimizer'] + op_function

        stats['learning_rate'] = args['lr']
        stats['batch_size'] = args['batch_size']
        all_stats.append(stats)

stats_df = pd.concat(all_stats)
stats_df.head()

stats_df = stats_df[stats_df['learning_rate'] >= 0.001]

# plot results
sns.set(palette="muted", font_scale=1.1)
sns.set_style("ticks")

# training loss
# # g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate',height=5, aspect=1)
g = sns.FacetGrid(stats_df, col="learning_rate", row='batch_size', hue='optimizer',height=8, aspect=1)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    plt.ylim(0, 0.5)
g = g.map(plot_loss, "epoch", "train_loss").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Training Loss (MNIST)", fontsize=15)
g.savefig("figures\mnist_lr_search_train_loss.png") 

# # training accuracy
# # g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=5, aspect=1.25)
g = sns.FacetGrid(stats_df, col="learning_rate", row='batch_size', hue='optimizer',height=8, aspect=1)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    plt.ylim(0.94, 1)
g = g.map(plot_loss, "epoch", "train_acc").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Training Accuracy (MNIST)", fontsize=15)
g.savefig("figures\mnist_lr_search_train_acc.png") 

# # validation accuracy
# # g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=5, aspect=1.25)
g = sns.FacetGrid(stats_df, col="learning_rate", row='batch_size', hue='optimizer',height=8, aspect=1)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    plt.ylim(0.9, 1)
g = g.map(plot_loss, "epoch", "val_acc").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Validation Accuracy (MNIST)", fontsize=15)
g.savefig("figures\mnist_lr_search_val_acc.png") 


# used to get the csv files for information for tables

# df_copy = stats_df.copy()
# last_epoch = max(df_copy['epoch'])
# df_copy = df_copy[df_copy['epoch']== last_epoch]
# print("we print EXTRA's epoch 100: ")
# print(df_copy.to_string())

# os.makedirs('folder/subfolder', exist_ok=True)  
# df_copy.to_csv('csv_files/main_experiments_MNIST.csv')  