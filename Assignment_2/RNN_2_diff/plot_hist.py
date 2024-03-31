"""
Jimut B. Pal,  30th March, 2024.
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

plt.figure(figsize=(12,6))
# plt.title(f'$\\textbf{{Loss Fold = }}$ {fold_no+1}',fontsize=30)
plt.xlabel(r'$\textbf{Epochs} \rightarrow$',fontsize=20)
plt.ylabel(r'$\textbf{Binary Cross Entropy Loss} \rightarrow$',fontsize=20)

# make a list of 6 colors in hexadecimal format
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']

for fold_no, col_p in zip(range(5), colors):
    # Specify the file path
    file_path = "history/test_logs_fold_{}.txt".format(fold_no)  # replace 'your_file.txt' with the actual file path

    history_data = []
    with open(file_path, 'r') as file:
        history_data = file.read()

    # Remove square brackets and split the string by commas
    history_data_cleaned = history_data.strip('[').replace(' ', '')
    history_data_cleaned = history_data_cleaned.strip('] \n').replace(' ', '')
    print(history_data_cleaned)
    history_list = [float(x) for x in history_data_cleaned.split(',')]
    
    plt.plot(history_list,'-', color=col_p,linewidth=3.0)
    plt.tight_layout()


file_path = "history/test_logs_final.txt"  # replace 'your_file.txt' with the actual file path

history_data = []
with open(file_path, 'r') as file:
    history_data = file.read()

# Remove square brackets and split the string by commas
history_data_cleaned = history_data.strip('[').replace(' ', '')
history_data_cleaned = history_data_cleaned.strip('] \n').replace(' ', '')
print(history_data_cleaned)
history_list = [float(x) for x in history_data_cleaned.split(',')]

plt.plot(history_list,'-', color=colors[-1],linewidth=3.0)
plt.tight_layout()


plt.legend([f'Fold = {i+1}' for i in range(5)]+['Final'], loc='upper right',fontsize=20)

plt.savefig("Train_Loss_full.png", bbox_inches='tight', dpi=500)
# plt.savefig("Train_Loss_full.eps", bbox_inches='tight', dpi=500)
plt.show()

