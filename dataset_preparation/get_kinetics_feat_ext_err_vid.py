import pandas as pd
import numpy as np
import glob as gb
from skvideo.io import ffprobe
import os
import pdb

def num_files_in_dir(path):
    return len(os.listdir(path))

def num_frames_in_vid(vid_path):
    metadata = ffprobe(vid_path)
    return int(metadata['video']['@nb_frames'])

# 1) read kinetics-7 annotation files
kd7_train_anno_path = '/net/acadia9a/data/jchoi/data/kinetics/anno/kinetics-7-train.csv'
df = pd.read_csv(kd7_train_anno_path, header=None)
data = df.to_numpy()

kd7_val_anno_path = '/net/acadia9a/data/jchoi/data/kinetics/anno/kinetics-7-val.csv'
df = pd.read_csv(kd7_val_anno_path, header=None)
data = np.concatenate([data, df.to_numpy()])

anno_vids = []
anno_video_path = {}
for row in data:
    vid = row[0].split('/')[-1].split('.')[0]
    anno_vids.append(vid)
    anno_video_path[vid] = os.path.join('/net/acadia6a/data/public_datasets/kinetics',row[0])

anno_vids = np.array(anno_vids)

# 2) get feature-extracted kinetics-7 directories
kd7_feat_dir = '/net/acadia9a/data/jchoi/data/kinetics/RGB-feature_resnet101'

feat_vids = os.listdir(kd7_feat_dir)
empty_vids = []
full_vids = []
empty_threshold = 10

for i,vid in enumerate(feat_vids):
    if i%1000 == 0:
        print('Processing dir {}/{}'.format(i+1,len(feat_vids)))
    cur_vid_path = os.path.join(kd7_feat_dir, vid)
    if num_files_in_dir(cur_vid_path) < empty_threshold:
        empty_vids.append(vid)
    else:
        full_vids.append(vid)
empty_vids = np.array(empty_vids)
full_vids = np.array(full_vids)
print('Done')

# 2)-1). see if features are extracted well
empty_vids = []
full_vids = []
num_frms_ext = []
num_frms_in_vid = []
for i,vid in enumerate(feat_vids):
    if i%1000 == 0:
        print('Processing dir {}/{}'.format(i+1,len(feat_vids)))
    cur_vid_path = os.path.join(kd7_feat_dir, vid)
    num_frms = num_files_in_dir(cur_vid_path)
    num_frms_ffprobe = num_frames_in_vid(anno_video_path[vid])
    num_frms_ext.append(num_frms)
    num_frms_in_vid.append(num_frms_ffprobe)
    if num_frms < num_frms_ffprobe:
        empty_vids.append(vid)
    else:
        full_vids.append(vid)
empty_vids = np.array(empty_vids)
full_vids = np.array(full_vids)
if np.array_equal(np.array(num_frms_ext), np.array(num_frms_in_vid)):
    print('Done verifying feature extraction')
else:
    print('There is something wrong with the feature extraction')
    pdb.set_trace()
    
# 3) get difference between 1 and 2
anno_vids = set(anno_vids)
full_vids = set(full_vids)

print('# of vids not in the feature extracted list but in the annotation list = {}'.format(len(anno_vids.difference(full_vids))))

print('Done verifying feat-ext-vid list and the anno-list')
# 4) Confirmed that the list of K-7 feature extracted videos is exactly same as the list of K-7 annotation list of videos (union of K7 train and K7 val videos)
# By the way there is an overlap between K7 train ad K7 val. The list of the vids are as follows:
# ['dLtsGCplNxU_000003_000013', 'MNKv6S5n_qg_000002_000012', 'zYGAQ3rPqbg_000037_000047', 'GAsDuNGVHH8_000004_000014', 'eLoI2XYD-gY_000148_000158', 'AiXtRGtyURY_000006_000016', 'mH77exACj00_000158_000168', '5Lvf39H7F-c_000004_000014', '3GJ1XCcksl0_000002_000012', '6Aw-FRzp2cs_000000_000010', 'Q0dbHju8wmo_000004_000014', '_h0cIb10KMQ_000000_000010', 'AGaCbzCtp1U_000010_000020', '6oxymlT4EU8_000000_000010', '2tqbNS1zRPI_000004_000014', 'axpsUkBY5AA_000002_000012', 'n1tEo4kbj-Y_000000_000010', 'H42ssq-EETA_000002_000012', 'R6o-RkFMTKE_000001_000011', 'JlVnYQeSzz8_000042_000052', '75RR6vzp4M4_000081_000091', 'OukJMZaXALA_000037_000047', 'bkWK79-Dm3A_000015_000025', 'aGkbmj6Te6w_000002_000012', 'Z7BDwuckFqE_000000_000010', 'ziB6UtxvwQA_000298_000308', 'E41cYR0WlZU_000008_000018', 'KBgJakE6MY4_000004_000014', 'PS40LDXED38_000000_000010', 'pxZmamAWSxU_000003_000013', 'n4EAEJ4xqpk_000001_000011', 'Xcz-Oho0tlU_000238_000248', 'ij5AVQNrJmw_000016_000026', 'tlGZSOtkGU0_000001_000011', 'qmP9d61ghDg_000001_000011', '9hNqrkHhFwE_000000_000010', 'p5VB0vPTU1g_000001_000011', 'VfoIi5dRSec_000097_000107', 'OafjzzoKir0_000003_000013', '1WxsQOhqc4M_000000_000010', 'pffxjlrD-2E_000002_000012', '-LU2saCAJY4_000397_000407', 'pFun05idyXk_000528_000538'] 