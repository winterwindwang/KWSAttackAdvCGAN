#!/usr/bin/env python
"""Train a CNN for Google speech commands."""

import argparse
import time
import os
from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch import optim
from center_loss import CenterLoss
import torchvision
from torchvision.transforms import *

# from tensorboardX import SummaryWriter
from pytorch_mfcc import MFCC
from utils import AverageMeter, Logger

import models
from datasets import *
import numpy as np
# from transforms import *
# from mixup import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp

def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    # plt.legend(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'], loc='upper right')
    plt.legend(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'], bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, dpi=300,  bbox_inches='tight')
    plt.close()

def train(epoch):
    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    mfcc_layer.eval()
    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()
    if args.plot:
        all_features, all_labels = [], []

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for input, label in pbar:
        # print(batch)
        # print(batch['mel_spectrogram'].shape)  # the shape was equal to input shape
        input = torch.unsqueeze(input, 1)


        lengths = [input.size(2) for _ in range(input.size(0))]
        val, mfcc_lengths = mfcc_layer(torch.squeeze(input.detach()), lengths)
        # inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)
        inputs = Variable(val, requires_grad=True)
        targets = Variable(label, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True)

        # forward/backward
        inputs = torch.unsqueeze(inputs, dim=1)
        # outputs is the original, output1, feature is the centerloss examples
        outputs, features, output1 = model(inputs)
        # loss_xent = criterion(outputs, targets)
        loss_xent = criterion(output1, targets)
        loss_cent = criterion_cent(features, targets)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizer.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()
        losses.update(loss.item(), targets.size(0))
        xent_losses.update(loss_xent.item(), targets.size(0))
        cent_losses.update(loss_cent.item(), targets.size(0))
        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(targets.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(targets.data.numpy())
        # statistics
        it += 1
        global_step += 1
        # print(loss.item())
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)


        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100 * correct / total)
        })
    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, 10, epoch, prefix='train')
    accuracy = correct / total
    epoch_loss = running_loss / it


def valid(epoch):
    global best_accuracy, best_loss, global_step

    phase = 'valid'
    model.eval()  # Set model to evaluate mode
    mfcc_layer.eval()
    running_loss = 0.0
    it = 0
    correct = 0
    total = 0
    if args.plot:
        all_features, all_labels = [], []
    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for input, label in pbar:

        input = torch.unsqueeze(input, 1)
        lengths = [input.size(2) for _ in range(input.size(0))]
        val, mfcc_lengths = mfcc_layer(torch.squeeze(input.detach()), lengths)

        # inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)
        inputs = Variable(val, requires_grad=True)
        targets = Variable(label, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True)
        inputs = torch.unsqueeze(inputs, dim=1)
        # outputs is the original, output1, feature is the centerloss examples
        outputs, features, output1 = model(inputs)
        # loss_xent = criterion(inputs, targets)
        loss_xent = criterion(output1, targets)
        loss_cent = criterion_cent(features, targets)
        loss = loss_cent + loss_xent

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100 * correct / total)
        })
        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(targets.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(targets.data.numpy())
    accuracy = correct.item() / total
    epoch_loss = running_loss / it
    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, 10, epoch, prefix='test')
    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer': optimizer.state_dict(),
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s_adv_tf.pth' % full_name)
        torch.save(model, 'runs/model/%d-%s-best-loss_adv_tf.pth' % (start_timestamp, full_name))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, 'checkpoints/best-acc-speech-commands-checkpoint-%s_adv_tf.pth' % full_name)
        torch.save(model, 'runs/model/%d-%s-best-acc_adv_10_tf.pth' % (start_timestamp, full_name))

    torch.save(checkpoint, 'checkpoints/last-speech-commands-checkpoint_adv_tf.pth')
    del checkpoint  # reduce memory

    return epoch_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-dataset", type=str, default=r'E:\exp\speech_commands\train', help='path of train dataset')
    parser.add_argument("--valid-dataset", type=str, default=r'E:\exp\speech_commands\valid', help='path of validation dataset')
    parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
    parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
    parser.add_argument("--batch-size", type=int, default=128, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
    parser.add_argument("--weight-decay", type=float, default=1e-3, help='weight decay')
    parser.add_argument("--optim", choices=['sgd', 'adam'], default='adam', help='choices of optimization algorithms')
    parser.add_argument("--learning-rate", type=float, default=1e-3, help='learning rate for optimization')
    parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
    parser.add_argument("--max-epochs", type=int, default=150, help='max number of epochs')
    parser.add_argument("--resume", type=str, default=False, help='checkpoint file to resume')
    parser.add_argument("--model", choices=models.available_models, default=models.available_models[5], help='model of NN')
    parser.add_argument("--input", choices=['mel32'], default='mel40', help='input of NN')
    parser.add_argument('--mixup', action='store_true', help='use mixup')
    parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
    parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
    parser.add_argument('--plot', type=bool, default=True, help="whether to plot features for every epoch")
    parser.add_argument('--save-dir', type=str, default='data')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    n_mels = 32
    if args.input == 'mel40':
        n_mels = 40

    train_dataset = SpeechCommandsDataset_classifier(args.train_dataset, transform=transforms.ToTensor())
    valid_dataset = SpeechCommandsDataset_classifier(args.valid_dataset, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_gpu,
                              num_workers=args.dataload_workers_nums,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, pin_memory=use_gpu,
                              num_workers=args.dataload_workers_nums,drop_last=True)

    # a name used to save checkpoints etc.
    full_name = '%s_%s_bs%d_lr%.1e_wd_adv_10_classes_mel_new' % (args.model, args.optim, args.batch_size, args.learning_rate)
    if args.comment:
        full_name = '%s_%s' % (full_name, args.comment)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mfcc_layer = MFCC(winlen=0.02, winstep=0.032, numcep=26).to(device)  # MFCC layer
    # mfcc_layer = MFCC(winlen=0.03,numcep=40, nfilt=40).to(device)  # MFCC layer (1, 98, 40)
    model = models.create_model(model_name=args.model, num_classes=len(CLASSES_ALL), in_channels=1)

    criterion = torch.nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=10, feat_dim=2, use_gpu=use_gpu)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()


    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_timestamp = int(time.time()*1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    global_step = 0

    if args.resume:
        print("resuming a checkpoint '%s'" % args.resume)
        checkpoint = torch.load('checkpoints/last-speech-commands-checkpoint_adv_tf.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

        del checkpoint  # reduce memory

    if args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    print("training %s for Google speech commands..." % args.model)
    since = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        if args.lr_scheduler == 'step':
            lr_scheduler.step()

        train(epoch)
        epoch_loss = valid(epoch)

        if args.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)

        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
    print("finished")

