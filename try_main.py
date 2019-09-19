# from config import _C as cfg
import argparse
import os, sys
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from models import VGG_line
from dataset import Wireframe

from tqdm import tqdm
from tensorboardX import SummaryWriter
from function import weights_init, loadVGG16pretrain, optim_load, Dice_Loss, normal_cross_entropy_loss, weighted_cross_entropy_loss
from utils import progbar, save_model


def main():
    parser = argparse.ArgumentParser(description='Line Segment Detection (Training)')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=False,
                        default="VGG_train.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epoch", dest="epoch", default=-1, type=int)
    args = parser.parse_args()
   
    print('Available GPU number is {}'.format(torch.cuda.device_count()))


    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    THIS_DIR = os.path.abspath(os.curdir)
    print(THIS_DIR)
    save_dir = os.path.join(THIS_DIR, 'save')
    log_dir = os.path.join(THIS_DIR, 'log1')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    batchsize = 6
    learning_rate = 0.000001
    step_size = 350

    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1]))
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = VGG_line(device).to(device)
    model = VGG_line()
    #model.apply(weights_init)
    #loadVGG16pretrain(model)
    #torch.save(model.state_dict(), os.path.join(save_dir, 'model_minloss.pth'))
    start_epoch = 0
    end_epoch = 1000
    minloss = 1e30

    optimizer = optim_load(model, lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    if start_epoch > 0:
        model_path = os.path.join(save_dir, 'model_minloss.pth')
        optimState_path = os.path.join(save_dir, 'optimState_minloss.pth')
        model.load_state_dict(torch.load(model_path))
        print('model get')
        try:
            optimizer.load_state_dict(torch.load(optimState_path, map_location='cpu'))
            print('optimizer get')
        except:
            pass

    #if torch.cuda.device_count() > 1:
    devices = [0, 1]
    model = torch.nn.DataParallel(model, device_ids=devices).cuda()
    #else:
    #model = model.to(device)
   

    root_train = './train'
    #root_valid = '/data/line-data/EELine/valid'
    train_dataset = Wireframe(root=root_train, split='train')
    #val_dataset = Wireframe(root=root_valid, split='val')
    train_loader = DataLoader(train_dataset, batch_size=batchsize, num_workers=8, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batchsize, num_workers=8, shuffle=False)

    # Log
    writer = SummaryWriter(log_dir)

    # Loss: function Dice_loss, weight_loss, normal loss
    loss_weight = 0.001  # noraml


    for epoch in range(start_epoch, end_epoch):
        #train_epoch_loss = train(epoch, train_loader, model, optimizer, scheduler, device, loss_weight, writer)
        #if train_epoch_loss < minloss:
        #    minloss = train_epoch_loss
        #    save_model(save_dir, minloss, model, optimizer)
         #   print(minloss)

        train_epoch_loss = train(epoch, train_loader, model, optimizer, scheduler, loss_weight, writer)


def train(epoch, dataloader, model, optimizer, scheduler, loss_weight, writer):
    model.train()
    scheduler.step(epoch=epoch)
    print('\n Training AT epoch = {}'.format(epoch))
    print('current learning rate = {}\n'.format(scheduler.get_lr()))
    str_train = 'train'
    bar = progbar(len(dataloader), width=10)

    avgLoss = [0., 0.,0.,0.,0.,0.]
    for i, (images, labels) in enumerate(dataloader):
        #images, labels = images.to(device), labels.to(device)
        images, labels = images.cuda(), labels.cuda()

        preds_list = model(images)

        #preds_list = preds_lis[0]
        #batch_loss = [(Dice_Loss(preds, labels) + loss_weight * normal_cross_entropy_loss(preds, labels))/ (preds_list[0].shape[0]) for preds in preds_list]
        #batch_loss = [weighted_cross_entropy_loss(preds, labels)/ (preds_list[0].shape[0]) for preds in preds_list]
        batch_loss = []
        #batch_loss = torch.zeros((6, images.shape[0]))
        for preds in preds_list:
            #print(preds.max(), preds.min())
            loss1 = Dice_Loss(preds, labels) #.to(torch.device('cuda:1'))
            loss2 = normal_cross_entropy_loss(preds, labels) #.to(torch.device('cuda:1'))
            batch_loss.append(loss1 + loss_weight * loss2) #/ preds.shape[0])
        #loss = batch_loss[0].mean() + batch_loss[1].mean() + batch_loss[2].mean() +batch_loss[3].mean() +batch_loss[4].mean() +batch_loss[5].mean()#torch.sum(batch_loss)  
        loss = (batch_loss[0]+ batch_loss[1]+ batch_loss[2]+ batch_loss[3]+ batch_loss[
            4] + batch_loss[5]) #.to(torch.device('cuda:1'))
        #batch_loss = normal_cross_entropy_loss(preds_list, labels)/ preds_list.shape[0]
        #loss = batch_losss
        optimizer.zero_grad()
        #loss1.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        bar.update(i, [('1_loss', batch_loss[0]),('2_loss', batch_loss[1]),('3_loss', batch_loss[2]),
             ('4_loss', batch_loss[3]),('5_loss', batch_loss[4]),('fuse_loss', batch_loss[5])])
        avgLoss =[(avgLoss[k] * i + batch_loss[k].item()) / (i + 1) for k in range(len(avgLoss))]
        if i % 20 == 0:
            pic = torchvision.utils.make_grid(images[:8], nrow=8, padding=2)
            writer.add_image('img', pic)
            pic1 = torchvision.utils.make_grid(preds_list[0][:8], nrow=8, padding=2)
            pic2 = torchvision.utils.make_grid(preds_list[1][:8], nrow=8, padding=2)
            pic3 = torchvision.utils.make_grid(preds_list[2][:8], nrow=8, padding=2)
            pic4 = torchvision.utils.make_grid(preds_list[3][:8], nrow=8, padding=2)
            pic5 = torchvision.utils.make_grid(preds_list[4][:8], nrow=8, padding=2)
            pic6 = torchvision.utils.make_grid(preds_list[5][:8], nrow=8, padding=2)
            writer.add_image(str_train + '/pred_1', pic1)
            writer.add_image(str_train + '/pred_2', pic2)
            writer.add_image(str_train + '/pred_3', pic3)
            writer.add_image(str_train + '/pred_4', pic4)
            writer.add_image(str_train + '/pred_5', pic5)
            writer.add_image(str_train + '/fuse', pic6)
            la = torchvision.utils.make_grid(labels[:8], nrow=8, padding=2)
            writer.add_image('lab', la)

    log = '\n * Finished epoch # %d   ' \
              'Loss_1: %1.4f, Loss_2: %1.4f, Loss_3: %1.4f, Loss_4: %1.4f, Loss_5: %1.4f, fuse_loss: %1.4f\n' % (
              epoch, avgLoss[0], avgLoss[1], avgLoss[2], avgLoss[3], avgLoss[4], avgLoss[5])
    print(log)
    writer.add_scalar(str_train + '/loss_1', avgLoss[0])
    writer.add_scalar(str_train + '/loss_2', avgLoss[1])
    writer.add_scalar(str_train + '/loss_3', avgLoss[2])
    writer.add_scalar(str_train + '/loss_4', avgLoss[3])
    writer.add_scalar(str_train + '/loss_5', avgLoss[4])
    writer.add_scalar(str_train + '/fuse_loss', avgLoss[5])

    return avgLoss[0]+avgLoss[1]+avgLoss[2]+avgLoss[3]+avgLoss[4]+avgLoss[5]


def valid(epoch, dataloader, model, optimizer, scheduler, loss_weight, writer):
    model.eval()
    str_train = 'val'
    bar = progbar(len(dataloader), width=10)
    with torch.no_grad():
        avgLoss = [0., 0.,0.,0.,0.,0.]
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            preds_list = model(images)
            batch_loss = [(Dice_Loss(preds, labels) + loss_weight * normal_cross_entropy_loss(preds, labels))/ (preds_list[0].shape[0]) for preds in preds_list]
            #batch_loss = [weighted_cross_entropy_loss(preds, labels)/ (preds_list[0].shape[0]) for preds in preds_list]

            bar.update(i, [('1_loss', batch_loss[0]),('2_loss', batch_loss[1]),('3_loss', batch_loss[2]),
             ('4_loss', batch_loss[3]),('5_loss', batch_loss[4]),('fuse_loss', batch_loss[5])])
            avgLoss =[(avgLoss[k] * i + batch_loss[k].item()) / (i + 1) for k in range(len(avgLoss))]
            if i % 10 == 0:
                pic1 = torchvision.utils.make_grid(preds_list[0][:8], nrow=8, padding=2)
                pic2 = torchvision.utils.make_grid(preds_list[1][:8], nrow=8, padding=2)
                pic3 = torchvision.utils.make_grid(preds_list[2][:8], nrow=8, padding=2)
                pic4 = torchvision.utils.make_grid(preds_list[3][:8], nrow=8, padding=2)
                pic5 = torchvision.utils.make_grid(preds_list[4][:8], nrow=8, padding=2)
                writer.add_image(str_train + '/pred_1', pic1)
                writer.add_image(str_train + '/pred_2', pic2)
                writer.add_image(str_train + '/pred_3', pic3)
                writer.add_image(str_train + '/pred_4', pic4)
                writer.add_image(str_train + '/pred_5', pic5)
                pic = torchvision.utils.make_grid(images[:8], nrow=8, padding=2)
                writer.add_image('img', pic)
                la = torchvision.utils.make_grid(labels[:8], nrow=8, padding=2)
                writer.add_image('lab', la)

        log = '\n * Finished epoch # %d   ' \
              'Loss_1: %1.4f, Loss_2: %1.4f, Loss_3: %1.4f, Loss_4: %1.4f, Loss_5: %1.4f, fuse_loss: %1.4f\n' % (
              epoch, avgLoss[0], avgLoss[1], avgLoss[2], avgLoss[3], avgLoss[4], avgLoss[5])
        print(log)
        writer.add_scalar(str_train + '/loss_1', avgLoss[0])
        writer.add_scalar(str_train + '/loss_2', avgLoss[1])
        writer.add_scalar(str_train + '/loss_3', avgLoss[2])
        writer.add_scalar(str_train + '/loss_4', avgLoss[3])
        writer.add_scalar(str_train + '/loss_5', avgLoss[4])
        writer.add_scalar(str_train + '/fuse_loss', avgLoss[5])

    return avgLoss[0]+avgLoss[1]+avgLoss[2]+avgLoss[3]+avgLoss[4]+avgLoss[5]

if __name__ == "__main__":
    main()
