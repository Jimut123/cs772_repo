import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib

# plt.style.use("dark_background")

# print(plt.rcParams.keys())
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

plt.rc('font', family='serif')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath,amsfonts,bbm}"
plt.rc('axes', linewidth=1)
plt.rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'




######### Code to calculate metrics and stuffs #########
def get_metrics(ALL_LABELS, ALL_PREDS, FileName):

    # Calculate accuracy
    accuracy = accuracy_score(ALL_LABELS, ALL_PREDS)

    # Calculate precision
    precision = precision_score(ALL_LABELS, ALL_PREDS, average='macro')

    # Calculate recall
    recall = recall_score(ALL_LABELS, ALL_PREDS, average='macro')

    # Calculate F1-score
    f1 = f1_score(ALL_LABELS, ALL_PREDS, average='macro')

    accuracy = float(accuracy)*100
    precision = float(precision)*100
    recall = float(recall)*100
    f1 = float(f1)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    with open("Metrics_{}.txt".format(FileName), 'a') as f:
        # Text, Original GT, Predicted Label
        f.write("Accuracy: "+str(accuracy)+" \n")
        f.write("Precision: "+str(precision)+" \n")
        f.write("Recall: "+str(recall)+" \n")
        f.write("F1-score: "+str(f1)+" \n")
        f.close() 




######### Code to draw the confusion metrics and stuffs #########
def draw_cm(ALL_LABELS, ALL_PREDS, labels, FileName):
    # Compute confusion matrix
    cm = confusion_matrix(ALL_LABELS, ALL_PREDS)

    # Define class labels
    # 0 for hate, 1 for normal, and 2 for offense.

    

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(r'\bf{Confusion Matrix}', fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=20, fontweight=200)

    plt.tight_layout()
    plt.ylabel(r'\bf{True label}', fontsize=20)
    plt.xlabel(r'\bf{Predicted label}', fontsize=20)

    # Save the plot
    plt.savefig('cm_{}.png'.format(FileName), bbox_inches='tight', pad_inches=0.1, dpi=300)

    # Show plot
    plt.show()


############ using one_shot_7b_filename

ALL_TEXTS = []
ALL_LABELS = []
ALL_PREDS = []

data_array = np.load('xlnet_results.npy', allow_pickle=True)

print(data_array)

for pred_val in data_array['pred']:
    # print(pred_val)
    ALL_PREDS.append(pred_val)

for gt_val in data_array['gt']:
    # print(gt_val)
    ALL_LABELS.append(gt_val)  


classes = [r'\bf{Hate}', r'\bf{Normal}', r'\bf{Offense}']
draw_cm(ALL_LABELS, ALL_PREDS, classes, FileName='xlnet_saikat')
get_metrics(ALL_LABELS, ALL_PREDS, FileName='xlnet_saikat')
