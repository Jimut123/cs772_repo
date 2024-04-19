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


######### Code to read the file and stuffs #########
def read_file(filename, ALL_TEXTS, ALL_LABELS, ALL_PREDS):
    err_pred = 0
    # dummy_check = 1
    try:
        with open(filename, 'r') as file:
            for line in file:
                print(line)
                print("--")
                if 'Text = ' in line:
                    
                    # Clean the prediction first
                    text_only_pred_first_split = str(line).split("Prediction = ")[1]
                    print("Prediction ==>>> ",text_only_pred_first_split)

                    
                    if "hate,0" in text_only_pred_first_split:
                        pred = 0
                        print("Prediction == ",pred)

                    if "normal,1" in text_only_pred_first_split:
                        pred = 1
                        print("Prediction == ",pred)
                    
                    if "offense,2" in text_only_pred_first_split:
                        pred = 2
                        print("Prediction == ",pred)
                    
                    if 'I' in str(text_only_pred_first_split):
                        pred = 2
                        print("Prediction == ",pred)
                        # err_pred += 1
                        # print("Error Prediction == ",text_only_pred_first_split)
                        # continue 


                    text_only_first_split = str(line).split("Text = ")[1]
                    text_only_sec_split = text_only_first_split.split(" Label = ")[0]
                    text_only_sec_split = text_only_sec_split[1:-1]
                    print(text_only_sec_split)
                    

                    text_only_label_first_split = str(line).split("Label = ")[1]
                    text_only_label_sec_split = text_only_label_first_split.split(" Prediction = ")[0]
                    print("Label ==>>> ",text_only_label_sec_split)
                    
                    # if len(pred) > 1:
                    #     print("??????? ", pred)
                    #     dummy_check = 0
                    if text_only_sec_split in ALL_TEXTS:
                        print("Already present.")
                        continue
                    ALL_TEXTS.append(str(text_only_sec_split))
                    ALL_PREDS.append(pred)
                    ALL_LABELS.append(int(text_only_label_sec_split))

    except FileNotFoundError:
        print("File not found.")
    
    # if dummy_check == 0:
    #     print("FFF")
    
    return ALL_TEXTS, ALL_LABELS, ALL_PREDS




# Check all the filenames

one_shot_7b_filename = "one_shot_output_llama-2-7b-chat.txt"
one_shot_13b_filename = "one_shot_output_llama-2-13b-chat.txt"

five_shot_7b_filename = "five_shot_output_llama-2-7b-chat.txt"
five_shot_13b_filename = "five_shot_output_llama-2-13b-chat.txt"

############ using one_shot_7b_filename

ALL_TEXTS = []
ALL_LABELS = []
ALL_PREDS = []
ALL_TEXTS, ALL_LABELS, ALL_PREDS = read_file(one_shot_7b_filename, ALL_TEXTS, ALL_LABELS, ALL_PREDS)


# print("ALL_TEXTS = ",ALL_TEXTS)
# print("ALL_LABELS = ",ALL_LABELS)
# print("ALL_PREDS = ",ALL_PREDS)

classes = [r'\bf{Hate}', r'\bf{Normal}', r'\bf{Offense}']
draw_cm(ALL_LABELS, ALL_PREDS, classes, FileName='one_shot_7b')
get_metrics(ALL_LABELS, ALL_PREDS, FileName='one_shot_7b')


############ using one_shot_13b_filename

ALL_TEXTS = []
ALL_LABELS = []
ALL_PREDS = []
ALL_TEXTS, ALL_LABELS, ALL_PREDS = read_file(one_shot_13b_filename, ALL_TEXTS, ALL_LABELS, ALL_PREDS)


# print("ALL_TEXTS = ",ALL_TEXTS)
# print("ALL_LABELS = ",ALL_LABELS)
# print("ALL_PREDS = ",ALL_PREDS)

classes = [r'\bf{Hate}', r'\bf{Normal}', r'\bf{Offense}']
draw_cm(ALL_LABELS, ALL_PREDS, classes, FileName='one_shot_13b')
get_metrics(ALL_LABELS, ALL_PREDS, FileName='one_shot_13b')




############ using five_shot_7b_filename

ALL_TEXTS = []
ALL_LABELS = []
ALL_PREDS = []
ALL_TEXTS, ALL_LABELS, ALL_PREDS = read_file(five_shot_7b_filename, ALL_TEXTS, ALL_LABELS, ALL_PREDS)


# print("ALL_TEXTS = ",ALL_TEXTS)
# print("ALL_LABELS = ",ALL_LABELS)
# print("ALL_PREDS = ",ALL_PREDS)

classes = [r'\bf{Hate}', r'\bf{Normal}', r'\bf{Offense}']
draw_cm(ALL_LABELS, ALL_PREDS, classes, FileName='five_shot_7b')
get_metrics(ALL_LABELS, ALL_PREDS, FileName='five_shot_7b')


############ using five_shot_13b_filename

ALL_TEXTS = []
ALL_LABELS = []
ALL_PREDS = []
ALL_TEXTS, ALL_LABELS, ALL_PREDS = read_file(five_shot_13b_filename, ALL_TEXTS, ALL_LABELS, ALL_PREDS)


# print("ALL_TEXTS = ",ALL_TEXTS)
# print("ALL_LABELS = ",ALL_LABELS)
# print("ALL_PREDS = ",ALL_PREDS)

classes = [r'\bf{Hate}', r'\bf{Normal}', r'\bf{Offense}']
draw_cm(ALL_LABELS, ALL_PREDS, classes, FileName='five_shot_13b')
get_metrics(ALL_LABELS, ALL_PREDS, FileName='five_shot_13b')