from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
from efficientnet_pytorch import EfficientNet
from datapreprocess.dataset_TOO import MyDataset_train, MyDataset_val
from sklearn.preprocessing import label_binarize
import numpy as np
from torch.utils.data.dataset import Dataset
import random

torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)  # 全部输出
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(47)

use_gpu = torch.cuda.is_available()
print(use_gpu)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FocalLoss(nn.Module):

    def __init__(self, device, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)
        self.eps = 1e-6

    #         self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').to(self.device)
        #         BCE_loss = self.BCE_loss(input, target)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()


# some parameters
for i in range(0, 5):
    print('FOLD:', i)
    batch_size = 32
    lr = 0.01
    momentum = 0.9
    num_epochs = 100
    input_size = 256
    num_class = 5
    fold = i
    net_name = 'efficientnet-b3'
    split_path = '/home/zhm/BT_final/datapreprocess/train_val_t_dir.pkl'
    save_dir = './TOO_models'
    results = './TOO_results/fold' + str(fold) + '/'


    def loaddata(split_path, fold, batch_size, set_name, shuffle):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {
            'train': MyDataset_train(split_path=split_path, fold=fold,
                                     transform=data_transforms['train']),
            'test': MyDataset_val(split_path=split_path, fold=fold,
                                  transform=data_transforms['test'])
        }

        dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                          batch_size=batch_size,
                                                          shuffle=shuffle, num_workers=8, pin_memory=True)
                           for x in [set_name]}
        data_set_sizes = len(image_datasets[set_name])

        return dataset_loaders, data_set_sizes


    def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
        lr = init_lr * (0.8 ** (epoch // lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer


    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_class).cuda()
    criterion = FocalLoss(device=device, gamma=2.).to(device)
    optimizer = optim.SGD((model.parameters()), lr=lr,
                          momentum=momentum, weight_decay=0.0004)

    train_loss = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        optimizer = exp_lr_scheduler(optimizer, epoch)
        dset_loaders, dset_sizes = loaddata(split_path=split_path, fold=fold, batch_size=batch_size,
                                            set_name='train', shuffle=True)
        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        count = 0
        correct_train = 0
        model.train()
        for data in dset_loaders['train']:
            inputs, labels = data
            labels = label_binarize(labels, classes=[i for i in range(num_class)])
            labels = torch.from_numpy(labels)
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            _, preds = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 30 == 0 or outputs.size()[0] < batch_size:
                print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()))
                train_loss.append(loss.item())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / dset_sizes
        epoch_loss = running_loss / dset_sizes

        print('TrainLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()
        running_loss_val = 0.0
        running_corrects_val = 0
        cont = 0
        outPre = []
        outLabel = []
        outPuts = []
        dset_loaders, dset_sizes = loaddata(split_path=split_path, fold=fold, batch_size=20, set_name='test',
                                            shuffle=False)
        with torch.no_grad():
            for data in dset_loaders['test']:
                inputs, labels = data
                labels = label_binarize(labels, classes=[i for i in range(num_class)])
                labels = torch.from_numpy(labels)
                labels = torch.squeeze(labels.type(torch.LongTensor))

                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

                _, preds = torch.max(outputs.data, 1)
                _, labels = torch.max(labels.data, 1)

                running_loss_val += loss.item() * inputs.size(0)
                running_corrects_val += torch.sum(preds == labels.data)

                cont += 1

            test_loss = running_loss_val / dset_sizes
            test_acc = running_corrects_val.double() / dset_sizes
            print('TestLoss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))

            if test_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    # save best model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/fold" + str(fold) + '_' + net_name + '.pth'
    torch.save(model, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
