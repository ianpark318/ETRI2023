import torch
import pandas as pd
from dataset import RPDataset, GpsDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':30,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':32,
    'SEED':42
}

test_df = pd.read_csv('./tiny_test_data.csv', index_col = 0)
test_df['img_path'] = test_df['img_path'].apply(lambda x : x.replace('./ETRI_data_RP_png', '../ETRIdata'))

RP_tfms = A.Compose([
    A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
    A.Normalize()
], p=1)

Gps_tfms = A.Compose([
    A.Resize(width=112, height=112),
    A.Normalize()
], p=1)

RP_test_dataset = RPDataset(df=test_df, rp_path_list=test_df['img_path'].values, label_list=test_df['action'].values, tfms=RP_tfms)
RP_test_loader = DataLoader(RP_test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
Gps_test_dataset = GpsDataset(df=test_df, lat_path_list=test_df['lat'].values, lon_path_list=test_df['lon'].values, label_list=test_df['action'].values, tfms=Gps_tfms)
Gps_test_loader = DataLoader(Gps_test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

def inference(model, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i, data in enumerate(zip(tqdm(RP_test_loader), Gps_test_loader)):
            data1, data2 = data
            images, labels = data1
            gps, _ = data2

            images = images.to(device)
            # gps = gps.to(device)
            # labels = labels.to(device)

            logit = model(images)
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

model = torch.load('save_model/0426_eff.pth')
preds = inference(model, device)
confusion_matrix = confusion_matrix(test_df['action'], preds, labels=[x for x in [5, 10, 12, 13, 14]])
plt.figure(figsize = (25,25))
plt.title('Confusion Matrix')

sns.heatmap(confusion_matrix, annot=True)

f1 = f1_score(test_df['action'], preds, average='micro')
print('F1-score: {0:.4f}'.format(f1))
y_true = test_df['action']
y_pred = preds
target_names = [str(x) for x in [5, 10, 12, 13, 14]]
print(classification_report(y_true, y_pred, target_names=target_names))
