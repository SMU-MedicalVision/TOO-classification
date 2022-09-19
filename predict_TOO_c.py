from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import pickle
import pandas as pd
from PIL import Image
from sklearn.preprocessing import label_binarize
import numpy as np
import random

torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)  # 全部输出
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


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
setup_seed(666)

use_gpu = torch.cuda.is_available()
print(use_gpu)

# some parameters
batch_size = 32
lr = 0.01
momentum = 0.9
num_epochs = 200
input_size = 256
num_class = 5
net_name = 'efficientnet-b3'
split_path = '/home/zhm/BT_final/datapreprocess/train_val_t_dir.pkl'
test_path = '/home/zhm/BT_final/datapreprocess/test_t_dir.pkl'
external_test_path = '/home/zhm/BT_final/datapreprocess/external_test_t_dir.pkl'
model_path = '/home/zhm/BT_final/TOO_models_c/efficientnet-b3.pth'
clinic_path = '/home/zhm/BT_final/Final_data.xlsx'
external_clinic_path = '/home/zhm/BT_final/external_data.xlsx'
out1 = './data/TOO_c_extest.xlsx'
out2 = './data/TOO_c_extest_.xlsx'


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
model = torch.load(model_path)
model.eval()

file = open(external_test_path, 'rb')
pkl_data = pickle.load(file)
clinic = pd.read_excel(external_clinic_path, sheet_name="all")
IDS = []
nums = []
pred1s = []
p0s = []
p1s = []
p2s = []
p3s = []
p4s = []
labels = []
for i in range(len(pkl_data)):
    data = pkl_data[i]
    ID = data['ID']
    print(ID)
    IDS.append(ID)
    clinic_data = clinic[clinic['X检查号码'] == ID]
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

    clinics = torch.tensor(
        [gender, age, location, fracture, leukocyte, congestion, swelling, fever, tenderness, dyskinesia, r_m],
        dtype=torch.float)

    DIR = data['DIR']
    num = int(DIR.split('.')[-2].split('/')[-1])
    nums.append(num)
    label = data['Label3']
    labels.append(label)
    image = np.load(DIR)
    image = image['roi']
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    image = Image.fromarray(image)
    image = data_transforms['test'](image)

    inputs = torch.unsqueeze(image, dim=0)
    clinics = torch.unsqueeze(clinics, dim=0)
    inputs, clinics = Variable(inputs.cuda()), Variable(clinics.cuda())
    with torch.no_grad():
        outputs = model(inputs, clinics)
        outputs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs.data, 1)

        outPut1 = outputs.data.cpu()
        pred1 = preds.data.cpu()
        pred1 = np.array(pred1)[0]
        pred1s.append(pred1)
        p0 = np.array(outPut1)[0][0]
        p0s.append(p0)
        p1 = np.array(outPut1)[0][1]
        p1s.append(p1)
        p2 = np.array(outPut1)[0][2]
        p2s.append(p2)
        p3 = np.array(outPut1)[0][3]
        p3s.append(p3)
        p4 = np.array(outPut1)[0][4]
        p4s.append(p4)

IDS = pd.DataFrame(IDS, columns=['ID'])
p0s = pd.DataFrame(p0s, columns=['P0'])
p1s = pd.DataFrame(p1s, columns=['P1'])
p2s = pd.DataFrame(p2s, columns=['P2'])
p3s = pd.DataFrame(p3s, columns=['P3'])
p4s = pd.DataFrame(p4s, columns=['P4'])
ps_o = pd.DataFrame(pred1s, columns=['P_O'])
label2s = pd.DataFrame(labels, columns=['label'])
nums = pd.DataFrame(nums, columns=['num'])
writer = pd.ExcelWriter(out1, engine='openpyxl')
IDS.to_excel(writer, sheet_name='Sheet1', startcol=0, index=False)
p0s.to_excel(writer, sheet_name='Sheet1', startcol=1, index=False)
p1s.to_excel(writer, sheet_name='Sheet1', startcol=2, index=False)
p2s.to_excel(writer, sheet_name='Sheet1', startcol=3, index=False)
p3s.to_excel(writer, sheet_name='Sheet1', startcol=4, index=False)
p4s.to_excel(writer, sheet_name='Sheet1', startcol=5, index=False)
ps_o.to_excel(writer, sheet_name='Sheet1', startcol=6, index=False)
label2s.to_excel(writer, sheet_name='Sheet1', startcol=7, index=False)
nums.to_excel(writer, sheet_name='Sheet1', startcol=8, index=False)
writer.save()
writer.close()


pics = pd.read_excel(out1, sheet_name='Sheet1')
ids = pics['ID']

IDs = []
p0s = []
p1s = []
p2s = []
p3s = []
p4s = []
# p5s = []
# p6s = []
# p7s = []
# p_ts = []
p_os = []
for i in range(len(pkl_data)):
    data = pkl_data[i]
    ID = data['ID']
    print(ID)
    t = pics[ids == ID].values.tolist()
    print(t)
    if len(t) == 1:
        ID = t[0][0]
        p0 = t[0][1]
        p1 = t[0][2]
        p2 = t[0][3]
        p3 = t[0][4]
        p4 = t[0][5]

        # p5 = t[0][8]
        # p6 = t[0][9]
        # p7 = t[0][10]
        p_o = t[0][6]
        # p_t = t[0][11]
    else:
        ID = t[0][0]
        p_o0s = []
        p_o1s = []
        p_o2s = []
        p_o3s = []
        p_o4s = []
        # p_o5s = []
        # p_o6s = []
        # p_o7s = []
        for j in range(len(t)):
            p_o0 = t[j][1]
            p_o1 = t[j][2]
            p_o2 = t[j][3]
            p_o3 = t[j][4]
            p_o4 = t[j][5]
            # p_o5 = t[j][8]
            # p_o6 = t[j][9]
            # p_o7 = t[j][10]

            p_o0s.append(p_o0)
            p_o1s.append(p_o1)
            p_o2s.append(p_o2)
            p_o3s.append(p_o3)
            p_o4s.append(p_o4)
            # p_o5s.append(p_o5)
            # p_o6s.append(p_o6)
            # p_o7s.append(p_o7)

        p0 = np.mean(p_o0s)
        p1 = np.mean(p_o1s)
        p2 = np.mean(p_o2s)
        p3 = np.mean(p_o3s)
        p4 = np.mean(p_o4s)
        # p5 = np.mean(p_o5s)
        # p6 = np.mean(p_o6s)
        # p7 = np.mean(p_o7s)
        if p0 > p1 and p0 > p2 and p0 > p3 and p0 > p4:
            p_o = 0
        elif p1 > p0 and p1 > p2 and p1 > p3 and p1 > p4:
            p_o = 1
        elif p2 > p0 and p2 > p1 and p2 > p3 and p2 > p4:
            p_o = 2
        elif p3 > p0 and p3 > p1 and p3 > p2 and p3 > p4:
            p_o = 3
        elif p4 > p0 and p4 > p2 and p4 > p3 and p4 > p1:
            p_o = 4

        # if p5 > p6 and p5 > p7:
        #     p_t = 0
        # elif p6 > p5 and p6 > p7:
        #     p_t = 1
        # elif p7 > p5 and p7 > p6:
        #     p_t = 2
    p0s.append(p0)
    p1s.append(p1)
    p2s.append(p2)
    p3s.append(p3)
    p4s.append(p4)

    # p5s.append(p5)
    # p6s.append(p6)
    # p7s.append(p7)
    p_os.append(p_o)
    # p_ts.append(p_t)
    IDs.append(ID)

IDS = pd.DataFrame(IDs, columns=['ID'])
p0s = pd.DataFrame(p0s, columns=['P0'])
p1s = pd.DataFrame(p1s, columns=['P1'])
p2s = pd.DataFrame(p2s, columns=['P2'])
p3s = pd.DataFrame(p3s, columns=['P3'])
p4s = pd.DataFrame(p4s, columns=['P4'])
ps_o = pd.DataFrame(p_os, columns=['P_O'])
# p5s = pd.DataFrame(p5s, columns=['P5'])
# p6s = pd.DataFrame(p6s, columns=['P6'])
# p7s = pd.DataFrame(p7s, columns=['P7'])
# ps_t = pd.DataFrame(p_ts, columns=['P_T'])
writer = pd.ExcelWriter(out2, engine='openpyxl')
IDS.to_excel(writer, sheet_name='Sheet1', startcol=0, index=False)
p0s.to_excel(writer, sheet_name='Sheet1', startcol=1, index=False)
p1s.to_excel(writer, sheet_name='Sheet1', startcol=2, index=False)
p2s.to_excel(writer, sheet_name='Sheet1', startcol=3, index=False)
p3s.to_excel(writer, sheet_name='Sheet1', startcol=4, index=False)
p4s.to_excel(writer, sheet_name='Sheet1', startcol=5, index=False)
ps_o.to_excel(writer, sheet_name='Sheet1', startcol=6, index=False)
# p5s.to_excel(writer, sheet_name='Sheet1', startcol=7, index=False)
# p6s.to_excel(writer, sheet_name='Sheet1', startcol=8, index=False)
# p7s.to_excel(writer, sheet_name='Sheet1', startcol=9, index=False)
# ps_t.to_excel(writer, sheet_name='Sheet1', startcol=10, index=False)
writer.save()
writer.close()

