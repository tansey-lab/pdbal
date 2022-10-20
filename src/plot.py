import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import os
import pickle


def load_results(folder):
    fnames = [x for x in os.listdir(folder) if (x.endswith('.pkl') and (os.path.getsize(os.path.join(folder, x)) > 0) )]
    num_exps = len(fnames)
    results = []
    for fname in fnames:
        with open(os.path.join(folder, fname), 'rb') as io:
            res = pickle.load(io)
        results.extend(res)
    
    df = pd.DataFrame(results)
    df['Query'] = df['Query'] + 1
    
    df.replace(to_replace="DBAL", value="PDBAL", inplace=True)
    return(df, num_exps)


colors = sns.color_palette("husl", 4)
orders = ['Random', 'Var', 'EIG', 'PDBAL']

palette={'Random':colors[0], 
         'Var':colors[1], 
         'EIG':colors[2], 
         'PDBAL':colors[3]}


def plot_folder(folder, ax, toprow=False, use_errorbar=True, keep_legend=False):
    df, num_exps = load_results(folder)
    errorbar = None
    if use_errorbar:
        factor = 1.645/np.sqrt(num_exps)
        errorbar = ('sd', factor)

    
    sns.lineplot(x='Query', y='Objective distance', hue='Strategy', style='Strategy', linewidth=2.5, errorbar=errorbar, data=df, palette=palette, hue_order=orders, ax=ax)
    if 'first' in folder:
        ax.set_ylabel("First coordinate error")
    elif 'max' in folder:
        ax.set_ylabel("Max coordinate error")
    elif 'euclidean' in folder:
        ax.set_ylabel("Euclidean error")
    elif 'kendall' in folder:
        ax.set_ylabel("Kendall tau error")
    elif 'influence' in folder:
        ax.set_ylabel("Influence error")
        
    if toprow:
        if 'linreg' in folder:
            ax.set_title("Linear regression")
        elif 'logreg' in folder:
            ax.set_title("Logistic regression")
        elif 'poisson' in folder:
            ax.set_title("Poisson regression")
        elif 'beta' in folder:
            ax.set_title("Beta regression")
    if keep_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)
    else:
        ax.get_legend().remove()
    return


matplotlib.rcParams.update({'font.size': 20, "axes.labelweight": "bold", "axes.titleweight": "bold"})
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))

# errorbar = None
col = 0
plot_folder("../results/linreg/first/", axs[0,col], toprow=True, keep_legend=True)
plot_folder("../results/linreg/max/", axs[1,col], toprow=False)
plot_folder("../results/linreg/kendall/", axs[2,col], toprow=False)

col+=1
plot_folder("../results/poisson/first/", axs[0,col], toprow=True)
plot_folder("../results/poisson/max/", axs[1,col], toprow=False)
plot_folder("../results/poisson/kendall/", axs[2,col], toprow=False)

col+=1
plot_folder("../results/beta/first/", axs[0,col], toprow=True)
plot_folder("../results/beta/max/", axs[1,col], toprow=False)
plot_folder("../results/beta/kendall/", axs[2,col], toprow=False)

plt.tight_layout()
plt.savefig("simul_fig.pdf", bbox_inches='tight')
plt.show()