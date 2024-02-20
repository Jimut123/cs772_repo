"""

Jimut Bahan Pal, 15th Feb, 2024.
10:47 P.M.
Code to plot the history of the training.

"""

import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

plt.rcParams['text.usetex'] = True
# print(plt.rcParams.keys())

# print(plt.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.spines.right'] = False
# matplotlib.rcParams['axes.spines.top'] = False

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath,amsfonts,bbm}"
plt.rc('axes', linewidth=1)
plt.rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

import seaborn as sns
sns.set_style("whitegrid")
sns.set(font_scale = 2)

for fold_no in range(4):
    # Specify the file path
    file_path = "train_logs_fold_{}.txt".format(fold_no)  # replace 'your_file.txt' with the actual file path

    # Initialize an empty list to store the data
    data = []

    # Read the file
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into tokens (assuming space-separated values)
            tokens = line.split()
            
            # Convert the tokens to the desired data type (e.g., integers or floats)
            loss_value = float(tokens[1])
            
            # Append the data to the list
            data.append(loss_value)

    # Display the read data
    print(data)

    plt.figure(figsize=(12,6))
    plt.title(f'$\\textbf{{Loss Fold = }}$ {fold_no+1}',fontsize=30)
    plt.xlabel(r'$\textbf{Epochs} \rightarrow$',fontsize=20)
    plt.ylabel(r'$\textbf{Binary Cross Entropy Loss} \rightarrow$',fontsize=20)
    plt.plot(data,'-', color='#0000FF',linewidth=3.0)
    # plt.tight_layout()
    plt.savefig("Train_Loss_Fold_{}.png".format(fold_no), bbox_inches='tight', dpi=500)
    # plt.savefig("Train_Loss_Fold_{}.eps".format(fold_no), bbox_inches='tight', dpi=500)
    # plt.show()

