from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  

def create_confusion_matrix(y_test, result, title, labels = ['BIG_COBBLE_STONE', 'SMALL_COBBLE_STONE', 'RUBBER_CURB', '']):
    cm = confusion_matrix(y_test, result)
#     print(cm)
    sum = np.sum(cm) 
    score = accuracy_score(y_test, result)
    
    precision_CLASS_A =  np.around(precision_score(y_test, result, average=None,pos_label=labels[0]), decimals=2)
#     precision_CLASS_A =  np.around(precision_score(y_test, result, average=None,pos_label=labels[1]), decimals=2)
#     precision_CLASS_A =  np.around(precision_score(y_test, result, average=None,pos_label=labels[2]), decimals=2)
    
    recall_CLASS_A = np.around(recall_score(y_test, result, average=None,pos_label=labels[0]), decimals=2)
#     recall_CLASS_A = np.around(recall_score(y_test, result, average=None,pos_label=labels[1]), decimals=2)
#     recall_CLASS_A = np.around(recall_score(y_test, result, average=None,pos_label=labels[2]), decimals=2)

    print('Precision: Class A',precision_CLASS_A)
    
    print('Recall: Class A',recall_CLASS_A)

    cm_new = np.append(cm[0], recall_CLASS_A[0])
    cm_new2 = np.append(cm[1], recall_CLASS_A[1])
    cm_new3 = np.append(cm[2], recall_CLASS_A[2])
    cm_new5 = np.array([precision_CLASS_A[0], precision_CLASS_A[1], precision_CLASS_A[2], score])
    cm = np.array([cm_new,cm_new2, cm_new3,cm_new5])
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, ax = ax,linewidths=.5,fmt='g',cmap="Greens"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title(title); 
    counter = 0
    for i in range(0,3):
        for j in range(0,4):
            percentage = cm[i,j]/sum
            t = ax.texts[counter]
            if j == 3:
                t.set_text(str(cm[i,j]))
            else:
                t.set_text(str(cm[i,j]) + '\n' + str(round(percentage*100,2)) + " %")
            counter = counter + 1

    ax.xaxis.set_ticklabels(labels, rotation='vertical')
    ax.yaxis.set_ticklabels(labels, rotation='horizontal')