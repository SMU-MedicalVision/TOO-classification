from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
from torchvision import datasets, models, transforms
import cv2
import pickle


class MyDataset_train(Dataset):
    def __init__(self, fold=0, split_path='/home/zhm/BT_final/datapreprocess/train_val_t_dir.pkl', transform=None):
        file = open(split_path, 'rb')
        pkl_data = pickle.load(file)
        if fold == -1:
            self.data_list = pkl_data[fold]["train"] + pkl_data[fold]['val']
        else:
            self.data_list = pkl_data[fold]["train"]
        self.length = len(self.data_list)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data_list[index]
        DIR = data['DIR']
        label2 = data['Label3']
        label = label2
        image = np.load(DIR)
        image = image['roi']
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.length


class MyDataset_val(Dataset):
    def __init__(self, fold=0, split_path='/home/zhm/BT_final/datapreprocess/train_val_t_dir.pkl', transform=None):
        file = open(split_path, 'rb')
        pkl_data = pickle.load(file)
        self.data_list = pkl_data[fold]['val']
        self.length = len(self.data_list)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data_list[index]
        ID = data['ID']
        DIR = data['DIR']
        label2 = data['Label3']
        label = label2
        image = np.load(DIR)
        image = image['roi']
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.length


class MyDataset_test(Dataset):
    def __init__(self, split_path='/home/zhm/BT_final/datapreprocess/test_t_dir.pkl', transform=None):
        file = open(split_path, 'rb')
        pkl_data = pickle.load(file)
        self.data_list = pkl_data
        self.length = len(self.data_list)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data_list[index]
        ID = data['ID']
        DIR = data['DIR']
        label2 = data['Label3']
        label = label2
        image = np.load(DIR)
        image = image['roi']
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.length


class MyDataset_train_c(Dataset):
    def __init__(self, fold=0, split_path='/home/zhm/BT_final/datapreprocess/train_val_t_dir.pkl',
                 clinic_path='/home/zhm/BT_final/Final_data.xlsx', transform=None):
        file = open(split_path, 'rb')
        pkl_data = pickle.load(file)
        if fold == -1:
            self.data_list = pkl_data[fold]["train"] + pkl_data[fold]['val']
        else:
            self.data_list = pkl_data[fold]["train"]
        self.length = len(self.data_list)
        self.transform = transform
        self.clinic = pd.read_excel(clinic_path, sheet_name="all")

    def __getitem__(self, index):
        data = self.data_list[index]
        ID = data['ID']
        DIR = data['DIR']
        label2 = data['Label3']
        label = label2
        image = np.load(DIR)
        image = image['roi']
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image)
        clinic_data = self.clinic[self.clinic['X检查号码'] == ID]

        gender = clinic_data['性别(男：1；女：0)'].values.tolist()[0]
        age = clinic_data['年龄（y）'].values.tolist()[0] / 100
        location = clinic_data['部位(中轴骨：0；四肢骨：1)'].values.tolist()[0]
        fracture = clinic_data['病理性骨折(0=无，1=有)'].values.tolist()[0]
        leukocyte = clinic_data['白细胞(异常：1；正常0)'].values.tolist()[0]
        congestion = clinic_data['充血'].values.tolist()[0]
        swelling = clinic_data['肿胀'].values.tolist()[0]
        fever = clinic_data['发热'].values.tolist()[0]
        tenderness = clinic_data['触痛'].values.tolist()[0]
        dyskinesia = clinic_data['运动障碍'].values.tolist()[0]
        r_m = clinic_data['可触及包块'].values.tolist()[0]

        clinic = torch.tensor(
            [gender, age, location, fracture, leukocyte, congestion, swelling, fever, tenderness, dyskinesia, r_m],
            dtype=torch.float)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, clinic

    def __len__(self):
        return self.length


class MyDataset_val_c(Dataset):
    def __init__(self, fold=0, split_path='/home/zhm/BT_final/datapreprocess/train_val_t_dir.pkl',
                 clinic_path='/home/zhm/BT_final/Final_data.xlsx', transform=None):
        file = open(split_path, 'rb')
        pkl_data = pickle.load(file)
        self.data_list = pkl_data[fold]['val']
        self.length = len(self.data_list)
        self.transform = transform
        self.clinic = pd.read_excel(clinic_path, sheet_name="all")

    def __getitem__(self, index):
        data = self.data_list[index]
        ID = data['ID']
        DIR = data['DIR']
        label2 = data['Label3']
        label = label2
        image = np.load(DIR)
        image = image['roi']
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image)
        clinic_data = self.clinic[self.clinic['X检查号码'] == ID]

        gender = clinic_data['性别(男：1；女：0)'].values.tolist()[0]
        age = clinic_data['年龄（y）'].values.tolist()[0] / 100
        location = clinic_data['部位(中轴骨：0；四肢骨：1)'].values.tolist()[0]
        fracture = clinic_data['病理性骨折(0=无，1=有)'].values.tolist()[0]
        leukocyte = clinic_data['白细胞(异常：1；正常0)'].values.tolist()[0]
        congestion = clinic_data['充血'].values.tolist()[0]
        swelling = clinic_data['肿胀'].values.tolist()[0]
        fever = clinic_data['发热'].values.tolist()[0]
        tenderness = clinic_data['触痛'].values.tolist()[0]
        dyskinesia = clinic_data['运动障碍'].values.tolist()[0]
        r_m = clinic_data['可触及包块'].values.tolist()[0]

        clinic = torch.tensor(
            [gender, age, location, fracture, leukocyte, congestion, swelling, fever, tenderness, dyskinesia, r_m],
            dtype=torch.float)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, clinic

    def __len__(self):
        return self.length


class MyDataset_test_c(Dataset):
    def __init__(self, split_path='/home/zhm/BT_final/datapreprocess/test_t_dir.pkl',
                 clinic_path='/home/zhm/BT_final/Final_data.xlsx', transform=None):
        file = open(split_path, 'rb')
        pkl_data = pickle.load(file)
        self.data_list = pkl_data
        self.length = len(self.data_list)
        self.transform = transform
        self.clinic = pd.read_excel(clinic_path, sheet_name="all")

    def __getitem__(self, index):
        data = self.data_list[index]
        ID = data['ID']
        DIR = data['DIR']
        label2 = data['Label3']
        label = label2
        image = np.load(DIR)
        image = image['roi']
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        image = Image.fromarray(image)
        clinic_data = self.clinic[self.clinic['X检查号码'] == ID]

        gender = clinic_data['性别(男：1；女：0)'].values.tolist()[0]
        age = clinic_data['年龄（y）'].values.tolist()[0] / 100
        location = clinic_data['部位(中轴骨：0；四肢骨：1)'].values.tolist()[0]
        fracture = clinic_data['病理性骨折(0=无，1=有)'].values.tolist()[0]
        leukocyte = clinic_data['白细胞(异常：1；正常0)'].values.tolist()[0]
        congestion = clinic_data['充血'].values.tolist()[0]
        swelling = clinic_data['肿胀'].values.tolist()[0]
        fever = clinic_data['发热'].values.tolist()[0]
        tenderness = clinic_data['触痛'].values.tolist()[0]
        dyskinesia = clinic_data['运动障碍'].values.tolist()[0]
        r_m = clinic_data['可触及包块'].values.tolist()[0]

        clinic = torch.tensor(
            [gender, age, location, fracture, leukocyte, congestion, swelling, fever, tenderness, dyskinesia, r_m],
            dtype=torch.float)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, clinic

    def __len__(self):
        return self.length


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_dataset = MyDataset_test_c(split_path='/home/zhm/BT_final/datapreprocess/test_t_dir.pkl',
                                     clinic_path='/home/zhm/BT_final/Final_data.xlsx', transform=None)
    print(train_dataset.__getitem__(index=1))
