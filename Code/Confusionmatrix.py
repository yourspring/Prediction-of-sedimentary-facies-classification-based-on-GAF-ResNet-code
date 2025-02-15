import os
import json
import pickle as pkl

import torch
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("The model accuracy is ", round(acc, 3))

        table = PrettyTable()
        table.field_names = ["Class", "Precision", "Recall", "Specificity", "F1-score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity, F1])
        print(table)

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig("ConfusionMatrix/ConfusionMatrix.jpg", dpi=300)
        plt.show()

if __name__ == '__main__':
    model_PATH = r'.\model\GAF-Resnet.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prediction = pkl.load(open("Datasets/prediction.pkl", "rb"))
    prediction_loader = data_utils.DataLoader(prediction, batch_size=1, shuffle=False)

    net = torch.load(model_PATH, weights_only=False).to(device)

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)

    net.eval()
    with torch.no_grad():
        for (x, y) in prediction_loader:
            input = x.to(device)

            if input.dim() == 3:
                input = input.unsqueeze(0)

            label = y.to(device)
            label_index = torch.argmax(label, dim=1)
            output = net(input)
            output = torch.argmax(output, dim=1)
            confusion.update(output.to("cpu").numpy().astype('int64'), label_index.to("cpu").numpy().astype('int64'))

    confusion.plot()
    confusion.summary()
