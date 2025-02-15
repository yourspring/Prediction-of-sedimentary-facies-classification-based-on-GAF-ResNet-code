import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import torch.utils.data as data_utils
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from net.Resnet18 import *
from Warm_up import *

PATH = r'.\model\GAF-Resnet.pt'

batch_size = 32
epoch_num = 80
lr = 0.00005

train = pkl.load(open("Datasets/train.pkl", "rb"))
validation = pkl.load(open("Datasets/validation.pkl", "rb"))
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
validation_loader = data_utils.DataLoader(validation, batch_size=batch_size, shuffle=False)

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


net = ResNet18(classes_num=5).to(device)

criterion = CrossEntropyLoss().to(device)

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)

total_steps = epoch_num * len(train_loader)
num_warmup_steps = int(0.1 * total_steps)
num_schedule_steps = int(total_steps - num_warmup_steps)
# scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
scheduler_coslr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_schedule_steps, eta_min=0.00001,
                                                       last_epoch=-1)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=num_warmup_steps,
                                          after_scheduler=scheduler_coslr)


train_accuracies = []
valid_accuracies = []

for epoch in range(epoch_num):
    net.train()
    tot_loss_train = 0.0
    correct_train = 0
    total_train = 0

    for (x, y) in train_loader:
        inputs = x.to(device)# (b,c,w,h)
        labels = y.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        scheduler_warmup.step()

        tot_loss_train += loss.item()

        _, labels_index = torch.max(labels, 1)
        _, pred_index = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (pred_index == labels_index).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    with open(r'Accuracy and loss/train_loss.txt', 'a') as f:
        f.write(str(tot_loss_train / len(train_loader)) + "\n")
    with open(r'Accuracy and loss/train_accuracy.txt', 'a') as f:
        f.write(str(train_accuracy) + "\n")

    net.eval()
    tot_loss_val = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for (x, y) in validation_loader:
            inputs = x.to(device)
            labels = y.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            tot_loss_val += loss.item()

            _, labels_index = torch.max(labels, 1)
            _, pred_index = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (pred_index == labels_index).sum().item()

    valid_accuracy = 100 * correct_val / total_val
    valid_accuracies.append(valid_accuracy)

    with open(r'Accuracy and loss/valid_loss.txt', 'a') as f:
        f.write(str(tot_loss_val / len(validation_loader)) + "\n")
    with open(r'Accuracy and loss/valid_accuracy.txt', 'a') as f:
        f.write(str(valid_accuracy) + "\n")

    lr_1 = optimizer.param_groups[0]['lr']
    print("learn_rate:%.8f" % lr_1)
    # schedule.step()
    print(epoch, tot_loss_train / len(train_loader), tot_loss_val / len(validation_loader))
    print('Accuracy of the network on the test datasets: %2d %%' % valid_accuracy)
    print('===========================================================')
    torch.save(net, PATH)

train_loss = np.genfromtxt("Accuracy and loss/train_loss.txt")
valid_loss = np.genfromtxt("Accuracy and loss/valid_loss.txt")
train_accuracy = np.genfromtxt("Accuracy and loss/train_accuracy.txt")
valid_accuracy = np.genfromtxt("Accuracy and loss/valid_accuracy.txt")

plt.figure()
plt.plot(train_loss, 'b-', label='Train_loss')
plt.plot(valid_loss, 'r-', label='Valid_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig("Accuracy and loss/Loss.jpg", dpi=300)
plt.show()

plt.figure()
plt.plot(train_accuracy, 'b-', label='Train_accuracy')
plt.plot(valid_accuracy, 'r-', label='Valid_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Valid Accuracy over Epochs')
plt.legend()
plt.savefig('Accuracy and loss/Accuracy.jpg', dpi=300)
plt.show()