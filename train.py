import torch
import time
import torch.nn as nn
from torch.nn import init
import os
import sys
from framework import Framework
from utils.datasets import prepare_Beijing_dataset, prepare_TLCGIS_dataset
from nets.DRENet import DRENet
import numpy as np
import random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print(torch.cuda.is_available())
torch.manual_seed(3407)
if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def get_model(model_name):
    if model_name == 'DRENet':
      model = DRENet(args.type)

    else:
        print("[ERROR] can not find model ", model_name)
        assert(False)
    return model

def get_dataloader(args):
    if args.dataset =='BJRoad':
        train_ds, val_ds, test_ds = prepare_Beijing_dataset(args) 
    elif args.dataset == 'deepglobe' or args.dataset.find('Porto') >= 0:
        train_ds, val_ds, test_ds = prepare_TLCGIS_dataset(args) 
    else:
        print("[ERROR] can not find dataset ", args.dataset)
        assert(False)  

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=True,  drop_last=False)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=True)
    return train_dl, val_dl, test_dl


def train_val_test(args):
    net = get_model(args.model)
    net = net.cuda()
    optimizer_config = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 6})

    
    optimizer = torch.optim.AdamW(net.parameters(), lr=optimizer_config['lr'], betas=optimizer_config['betas'], weight_decay=optimizer_config['weight_decay'])
    print(net)
    framework = Framework(net, optimizer, dataset=args.dataset)
    
    train_dl, val_dl, test_dl = get_dataloader(args)
    framework.set_train_dl(train_dl)
    framework.set_validation_dl(val_dl)
    framework.set_test_dl(test_dl)
    framework.set_save_path(WEIGHT_SAVE_DIR)

    framework.fit(epochs=args.epochs)


if __name__ == "__main__":
    import argparse
    #--epochs 50  --dataset "Porto"  --split_train_val_test "../dataset/porto_dataset/split"  --sat_dir "../dataset/porto_dataset/rgb"   --mask_dir "../dataset/porto_dataset/mask"    --lidar_dir "../dataset/porto_dataset/gps" --down_scale 'false'
    #--epochs 50  --dataset "deepglobe"  --split_train_val_test "../dataset/deepglobe/split"  --sat_dir "../dataset/deepglobe/sat"   --mask_dir "../dataset/deepglobe/mask"     --down_scale 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DRENet')
    parser.add_argument('--type', type=str, default='t')
    #parser.add_argument('--path', type=str, default='./mobilevit_s.pt')
    parser.add_argument('--lr',    type=float, default=8e-5)
    parser.add_argument('--name',  type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sat_dir',  type=str, default='../dataset/BJRoad/train_val/image')
    parser.add_argument('--mask_dir', type=str, default='../dataset/BJRoad/train_val/mask')
    parser.add_argument('--skeleton_dir', type=str, default='../dataset/BJRoad/train_val/skeleton')
    parser.add_argument('--gps_dir',  type=str, default='../dataset/BJRoad/train_val/gps')
    parser.add_argument('--test_sat_dir',  type=str, default='../dataset/BJRoad/test/image/')
    parser.add_argument('--test_mask_dir', type=str, default='../dataset/BJRoad/test/mask/')
    parser.add_argument('--test_gps_dir',  type=str, default='../dataset/BJRoad/test/gps/')
    parser.add_argument('--lidar_dir',  type=str, default='../dataset/porto_dataset/gps')
    parser.add_argument('--split_train_val_test', type=str, default='../dataset/porto_dataset/split')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model/')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=str, default='0')
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--dataset', type=str, default='BJRoad')
    parser.add_argument('--down_scale', type=bool, default=True)
    args = parser.parse_args()

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size * len(gpu_list)
    else:
        BATCH_SIZE = args.batch_size

    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir, f"{args.model}_{args.dataset}_"+time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+"/")
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.makedirs(WEIGHT_SAVE_DIR)
    print("Log dir: ", WEIGHT_SAVE_DIR)

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(WEIGHT_SAVE_DIR+'train.log')

    train_val_test(args)
    print("[DONE] finished")
    
    print("model:",args.model + "_" + args.type)
    print("batchsize:",BATCH_SIZE)
    print("lr:",args.lr)
    # os.system("shutdown")

