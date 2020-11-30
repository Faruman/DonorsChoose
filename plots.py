import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

# plot the results
# plot precision recall plot
def plot_auc(label, score, title):
    precision, recall, thresholds = precision_recall_curve(label, score)
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(thresholds, precision[1:], color='r', label='Precision')
    plt.plot(thresholds, recall[1:], color='b', label='Recall')
    plt.gca().invert_xaxis()
    plt.legend(loc='lower right')

    plt.xlabel('Threshold (0.00 - 1.00)')
    plt.ylabel('Precision / Recall')
    _ = plt.title(title)
    plt.savefig('./plots/prec_recall_curve_baselineModel.png')
    plt.show()