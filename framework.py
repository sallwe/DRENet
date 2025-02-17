import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from tqdm import tqdm
from utils.my_metrics import IoU
from transformers import AdamW, get_linear_schedule_with_warmup
from loss import *
import copy
import numpy
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR


class Solver:
    def __init__(self, net, optimizer, dataset):
        self.net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(torch.cuda.device_count())))
        
        self.optimizer = optimizer
        self.dataset = dataset
        len_data = len(dataset)
        epochs = 70
        batchsize = 4
        warmup_ratio = 0.1
        total_step = (len_data // batchsize) * epochs
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)

        self.loss = dice_bce_loss().cuda()
        self.focal_loss = FocalLoss().cuda()
        self.metrics = IoU(threshold=0.5)
        self.old_lr = optimizer.param_groups[0]["lr"]
        
    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch

    def data2cuda(self, volatile=False):
        if volatile:
            with torch.no_grad():
                self.img = Variable(self.img.cuda())
        else:
            self.img = Variable(self.img.cuda())

        if self.mask is not None:
            if volatile:
                with torch.no_grad():
                    self.mask = Variable(self.mask.cuda())
            else:
                self.mask = Variable(self.mask.cuda())

    def save_weights(self, path):
        torch.save(self.net.state_dict(), path)
    def load_weights(self, path):
        self.net.load_state_dict(torch.load(path,map_location='cpu'), strict=False)

    def optimize(self):
        self.net.train()
        self.data2cuda()
        
        self.optimizer.zero_grad()
        pred, skeleton_pred, sat_pred, pred_1 = self.net.forward(self.img)
        skeleton_mask = self.mask[:, 1:, :, :]
        mask = self.mask[:, 0:1, :, :]
        loss = self.loss(mask, pred) + self.loss(skeleton_mask, skeleton_pred) + self.loss(mask, sat_pred) + self.loss(mask, pred_1)
        # loss = self.loss(mask, sat_pred) + self.loss(mask, pred_1)
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()

        metrics = self.metrics(mask, pred)
        metrics_1 = self.metrics(mask, pred_1)
        return loss.item(),  metrics, metrics_1

    def test_batch(self):
        self.net.eval()
        self.data2cuda(volatile=True)
        skeleton_mask = self.mask[:, 1:, :, :]
        mask = self.mask[:, 0:1, :, :]
        #
        pred, skeleton_pred, sat_pred, pred_1 = self.net.forward(self.img)
        loss = self.loss(mask, pred) + self.loss(skeleton_mask, skeleton_pred) + self.loss(mask, sat_pred) + self.loss(mask, pred_1)
        # loss = self.loss(mask, sat_pred) + self.loss(mask, pred_1)

        #pred = self.net.forward(self.img)
        mask = self.mask[:, 0:1, :, :]
        # loss = self.loss(mask, pred)
        
        metrics = self.metrics(mask, pred)
        metrics_1 = self.metrics(mask, pred_1)

        return loss.item(),  metrics, metrics_1
        
        
    def update_lr(self, epoch, ratio=1.0):
        ratio = (50 - epoch)/ 50
        for param_group in self.optimizer.param_groups:
            new_lr = self.old_lr * ratio
            param_group["lr"] = new_lr
        print("==> update learning rate:  -> %f" % (new_lr))

    def pred_one_image(self, image):
        self.net.eval()
        # image = image.unsqueeze(0)
        pred, skeleton, sat_pred, pred_1 = self.net.forward(image)

        return pred.cpu().data.numpy().squeeze(1).squeeze(0), skeleton.cpu().data.numpy().squeeze(1).squeeze(0), pred_1.cpu().data.numpy().squeeze(1).squeeze(0)

class Framework:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)

    def set_train_dl(self, dataloader):
        self.train_dl = dataloader

    def set_validation_dl(self, dataloader):
        self.validation_dl = dataloader

    def set_test_dl(self, dataloader):
        self.test_dl = dataloader

    def set_save_path(self, save_path):
        self.save_path = save_path

    def fit(self, epochs, no_optim_epochs=5):
        test_best_metrics = 0.59
        no_optim = 0
        epoch_list, train_loss_list, val_loss_list, test_loss_list, train_metrics_list, val_metrics_list, test_metrics_list= list(),list(),list(),list(),list(),list(),list()
        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch}/{epochs}")

            train_loss, train_metrics, train_metrics_1 = self.fit_one_epoch(self.train_dl, mode='training')
            val_loss, val_metrics, val_metrics_1 = self.fit_one_epoch(self.validation_dl, mode='val')
            test_loss, test_metrics, test_metrics_1 = self.fit_one_epoch(self.test_dl, mode='testing')

            if test_metrics[3] > test_best_metrics:
                test_best_metrics = test_metrics[3]
                # self.solver.save_weights(os.path.join(self.save_path,
                #                                      f"epoch{epoch}_test_1{test_metrics_1[3]:.4f}_test{test_metrics[3]:.4f}.pth"))

            print(f'train_loss: {train_loss:.4f} train_metrics: {train_metrics}')
            train_loss_list.append(train_loss)
            train_metrics_list.append(float(train_metrics[3]))
            print(f'val_loss: {val_loss:.4f}   val_metrics:   {val_metrics}')
            val_loss_list.append(val_loss)
            val_metrics_list.append(float(val_metrics[3]))
            print(f'test_loss: {test_loss:.4f}  test_metrics:  {test_metrics} test_metrics_1:  {test_metrics_1}')
            test_loss_list.append(test_loss)
            test_metrics_list.append(float(test_metrics[3]))
            #self.solver.update_lr(epoch)
            
            print('epoch finished')
        loss_epoch = pd.DataFrame({"train_loss":train_loss_list,"val_loss":val_loss_list,"test_loss":test_loss_list},
                                  index=['epoch'+str(i) for i in range(1,epochs +1)])
        metric_epoch = pd.DataFrame({"train_metrics":train_metrics_list,"val_metrics":val_metrics_list,"test_metrics":test_metrics_list}
                                    ,index=['epoch'+str(i) for i in range(1,epoch + 1)])
        loss_epoch = loss_epoch.plot(kind='line')
        metric_epoch = metric_epoch.plot(kind='line')
        loss_epoch.figure.savefig(self.save_path+'loss_epoch_fig.png')
        metric_epoch.figure.savefig(self.save_path+'metric_epoch_fig.png')  


    def fit_one_epoch(self, dataloader, mode='training'):
        epoch_loss = 0
        epoch_metrics = 0.0
        epoch_metrics_1 = 0.0

        dataloader_iter = iter(dataloader)
        iter_num = len(dataloader_iter)
        progress_bar = tqdm(enumerate(dataloader_iter), total=iter_num)

        for i, (img, mask) in progress_bar:
            self.solver.set_input(img, mask)
            if mode=='training':
                iter_loss, iter_metrics, iter_metrics_1 = self.solver.optimize()
            else:
                iter_loss, iter_metrics, iter_metrics_1  = self.solver.test_batch()

            epoch_loss += iter_loss
            epoch_metrics += iter_metrics
            epoch_metrics_1 += iter_metrics_1
            progress_bar.set_description(f'{mode} iter: {i} loss: {iter_loss:.4f}')

        epoch_loss /= iter_num
        epoch_metrics /= iter_num
        epoch_metrics_1 /= iter_num
        return epoch_loss, epoch_metrics, epoch_metrics_1