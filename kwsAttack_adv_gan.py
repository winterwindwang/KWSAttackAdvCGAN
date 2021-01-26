import argparse
import time
import os
from tqdm import *
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.io import wavfile
import torch.nn as nn
from torch import optim
from pytorch_mfcc import MFCC
import torchvision

import numpy as np
import random
import models
from models.discriminator import CDiscriminator
from models.generator import CGenerator
from datasets import *
import torchnet.meter as meter
import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def wav_save(epoch, data, data_dir, label, target, name):
    singals = data.cpu().data.numpy()
    label = label.cpu().data.numpy()
    target = target.cpu().data.numpy()
    idx2classes = {0: 'yes', 1: 'no', 2: 'up', 3: 'down', 4: 'left', 5:'right',6:'on',7:'off',8:'stop',9:'go'}
    for i in range(len(singals)):
        output = singals[i].reshape(16384, 1)
        output = (output - 1) / (2 / 65535) + 32767
        output = output.astype(np.int16)
        labels = idx2classes[label[i]]
        dir = os.path.join(data_dir, labels)
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        filename = "{}_{}_to_{}_epoch_{}_{}.wav".format(name, idx2classes[label[i]], idx2classes[target[i]], epoch, i)
        path = os.path.join(dir, filename)
        wavfile.write(path, 16384, output)

def one_hot(label, num_class=10):
    '''
    :param batch_size:
    :param label: should be the shape [B,1], torch.squeeze(label,1)
    :param num_class:
    :return:
    '''
    label = torch.unsqueeze(label, dim=1)
    return torch.zeros(label.size(0), num_class).scatter(1, label, 1)

def train(epoch):
    global global_step
    f.eval()
    # f1.eval()
    # f2.eval()
    mfcc_layer.eval()
    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    # writer.add_scalar('%s/learning_rate' % phase, get_lr(), epoch)
    acc = 0
    epoch_loss_g = 0
    epoch_loss_d = 0
    n = 0
    correct = 0
    total = 0
    Tensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.LongTensor

    pbar = tqdm(train_loader, unit='audios', unit_scale=train_loader.batch_size)
    for inputs, orglabel in pbar:
        inputs = torch.unsqueeze(inputs, 1)
        # Batch Same
        # rnd_label = random.randint(0,9)
        # labels = torch.ones_like(orglabel).fill_(rnd_label)
        # Batch Diff
        labels = Variable(LongTensor(np.random.randint(0, 10, inputs.size(0))))
        valid = Variable(Tensor(inputs.size(0), 1).fill_(1.0), requires_grad=False)
        if use_gpu:
            inputs = inputs.to(device)
            labels = labels.to(device)
            valid = valid.to(device)
            orglabel = orglabel.to(device)

        g_optimizer.zero_grad()
        perturbation = torch.clamp(G(inputs, labels), -0.3, 0.3)
        adv_audio = perturbation + inputs
        fakes = adv_audio.clamp(-1., 1.)

        lengths = [inputs.size(2) for _ in range(inputs.size(0))]
        val, mfcc_lengths = mfcc_layer(torch.squeeze(fakes), lengths)
        pred1 = f(torch.unsqueeze(val, dim=1))
        # pred2 = f1(torch.unsqueeze(val, dim=1))
        # pred3 = f2(torch.unsqueeze(val, dim=1))
        # y_pred = (pred1 + pred2) / 2
        loss_adv = criterion_crossentropy(pred1, labels)
        # acc += torch.sum(torch.max(y_pred, 1)[1] == labels).item()
        # loss_adv += criterion_crossentropy(pred2,labels)
        # loss_adv += criterion_crossentropy(pred3,labels)
        outputs1 = torch.nn.functional.softmax(pred1, dim=1)
        # outputs2 = torch.nn.functional.softmax(pred2, dim=1)
        # outputs3 = torch.nn.functional.softmax(pred3, dim=1)
        predict1 = outputs1.data.max(1, keepdim=True)[1]
        # predict2 = outputs2.data.max(1, keepdim=True)[1]
        # predict3 = outputs3.data.max(1, keepdim=True)[1]
        pred = predict1 # (predict1 +  predict2 + predict3) / 3
        correct += pred.eq(labels.data.view_as(pred)).sum()
        total += labels.size(0)
        acc = correct.item() / total


        dg_fake,_ = D(fakes, labels)
        g_loss_d = criterion_gan(dg_fake, valid)
        g_loss_l1 = criterion_l2(fakes, inputs)
        g_loss = g_loss_d + loss_adv + 10000 * g_loss_l1 #+ loss_perturb

        g_loss.backward()
        g_optimizer.step()
        epoch_loss_g += g_loss.item()

        # Training Discriminator
        d_optimizer.zero_grad()
        d_real, _ =  D(inputs, orglabel)
        d_loss_real = torch.mean((d_real - 1.0) ** 2)  
        d_fake, _= D(fakes.detach(), labels)
        d_loss_fake = torch.mean(d_fake ** 2)
        d_loss =  (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        d_optimizer.step()

        epoch_loss_d += d_loss.item()
        global_step += 1
        n += inputs.size(0)

        loss_summar_train['loss_g'].append(g_loss.item())
        loss_summar_train['loss_gan'].append(g_loss_d.item())
        loss_summar_train['loss_adv'].append(loss_adv.item())
        loss_summar_train['loss_l2'].append(g_loss_l1.item())
        loss_summar_train['acc'].append(acc / n)
        # update the progress bar
        pbar.set_postfix({
            'd_loss': "%.05f" % (d_loss.mean().item()),
            'g_loss': "%.05f" % (g_loss.mean().item()),
            'g_loss_gan': "%.05f" % (g_loss_d.mean().item()),
            'loss_adv': "%.05f" % (loss_adv.mean()),
            'acc': "%.02f" % (acc )
        })
        if d_loss.item() == np.nan or g_loss.item() == np.nan:
            exit()
    accuracy = acc # / n
    epoch_loss_ds = epoch_loss_d / n
    epoch_loss_gs = epoch_loss_g / n
    np.save('loss_summar_with_hinge_train', loss_summar_train)

def valid(epoch):
    global best_accuracy, best_loss, global_step
    f.eval()
    # f1.eval()
    # f2.eval()
    G.eval()
    phase = 'valid'
    it = 0
    acc = 0
    n = 0
    epoch_loss_g = 0
    correct = 0
    total = 0
    Tensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    confusion_matrix = meter.confusionmeter.ConfusionMeter(10)

    LongTensor = torch.LongTensor
    pbar = tqdm(valid_loader, unit="audios", unit_scale=valid_loader.batch_size)
    for samples, orglabel in pbar:
        inputs = torch.unsqueeze(samples, 1)

        inputs = Variable(inputs, requires_grad=True)
        # Batch Same
        # rnd_label = random.randint(0,9)
        # labels = torch.ones_like(orglabel).fill_(rnd_label)
        # Batch Diff
        labels = Variable(LongTensor(np.random.randint(0, 10, inputs.size(0))))
        valid = Variable(Tensor(inputs.size(0), 1).fill_(1.0), requires_grad=False)

        if use_gpu:
            inputs = inputs.to(device)
            labels = labels.to(device)
            valid = valid.to(device)
            orglabel = orglabel.to(device)


        perturbation = torch.clamp(G(inputs,labels), -0.3, 0.3)
        adv_audio = perturbation + inputs
        fakes = adv_audio.clamp(-1., 1.)

        lengths = [inputs.size(2) for _ in range(inputs.size(0))]
        val, mfcc_lengths = mfcc_layer(torch.squeeze(fakes), lengths)
        # y_pred = f(torch.unsqueeze(val, dim=1))
        pred1 = f(torch.unsqueeze(val, dim=1))
        # pred2 = f1(torch.unsqueeze(val, dim=1))
        # pred3 = f2(torch.unsqueeze(val, dim=1))
        # y_pred = (pred1 + pred2) / 2
        loss_adv = criterion_crossentropy(pred1, labels)
        # acc += torch.sum(torch.max(y_pred, 1)[1] == labels).item()
        # loss_adv += criterion_crossentropy(pred2, labels)
        # loss_adv += criterion_crossentropy(pred3, labels)
        outputs1 = torch.nn.functional.softmax(pred1, dim=1)
        # outputs2 = torch.nn.functional.softmax(pred2, dim=1)
        # outputs3 = torch.nn.functional.softmax(pred3, dim=1)
        predict1 = outputs1.data.max(1, keepdim=True)[1]
        # predict2 = outputs2.data.max(1, keepdim=True)[1]
        # predict3 = outputs3.data.max(1, keepdim=True)[1]
        pred = predict1 #(predict1 + predict2 + predict3) / 3
        correct += pred.eq(labels.data.view_as(pred)).sum()
        total += labels.size(0)
        confusion_matrix.add(torch.squeeze(pred), labels.data)
        acc = correct.item() / total
        # loss_adv = criterion_crossentropy(y_pred, labels)
        # acc += torch.sum(torch.max(y_pred, 1)[1] == labels).item()

        dg_fake, _ = D(fakes, labels)
        g_loss_d = criterion_gan(dg_fake, valid)
        g_loss_l1 = criterion_l2(fakes, inputs)
        g_loss = g_loss_d + loss_adv + 10000 * g_loss_l1  # + loss_perturb
        epoch_loss_g += g_loss.item()
        # statistics
        n += labels.size(0)
        it += 1
        loss_summar_valid['loss_g'].append(g_loss.item())
        loss_summar_valid['loss_gan'].append(g_loss_d.item())
        loss_summar_valid['loss_adv'].append(loss_adv.item())
        loss_summar_valid['loss_l2'].append(g_loss_l1.item())
        loss_summar_valid['acc'].append(acc)

        # update the progress bar
        pbar.set_postfix({
            'g_loss': "%.05f" % (g_loss.mean().item()),
            'acc': "%.02f" % (acc )
        })
    accuracy = acc
    epoch_loss = epoch_loss_g / it

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': G.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, 'checkpoints/generator/best-acc-generator-checkpoint-%s.pth' % full_name)
        torch.save(G, 'runs/model/%d-%s-best-loss.pth' % (start_timestamp, full_name))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, 'checkpoints/generator/best-loss-generator-checkpoint-%s.pth' % full_name)
        torch.save(G, 'runs/model/%d-%s-best-acc.pth' % (start_timestamp, full_name))
        torch.save(checkpoint, 'checkpoints/generator/generator-checkpoint-epoch-%s.pth' % (epoch))
    torch.save(checkpoint, 'checkpoints/generator/last-generator-checkpoint.pth')
    torch.save(checkpoint, 'checkpoints/generator/generator-checkpoint-epoch-%d.pth'%(epoch))
    np.save('loss_summar_with_hinge_valid', loss_summar_valid)
    del checkpoint

    return accuracy

def test(epoch):
    pbar = tqdm(test_loader, unit="audios", unit_scale=test_loader.batch_size)
    LongTensor = torch.LongTensor
    for samples, orglabel in pbar:
        inputs = torch.unsqueeze(samples, 1)
        # Batch Same
        # rnd_label = random.randint(0,9)
        # labels = torch.ones_like(orglabel).fill_(rnd_label)
        # Batch Diff
        labels = Variable(LongTensor(np.random.randint(0, 10, inputs.size(0))))
        if use_gpu:
            inputs = inputs.to(device)
            labels = labels.to(device)

        perturbation = torch.clamp(G(inputs, labels), -0.3, 0.3)
        adv_audio = perturbation + inputs
        fakes = adv_audio.clamp(-1., 1.)
        # save the test data during training phrase
        targets = labels.cpu().data
        wav_save(epoch, fakes, 'samples/gen', orglabel, labels, 'fake')
        wav_save(epoch, perturbation, 'samples/pert', orglabel, labels, 'pert')
        wav_save(epoch, inputs, 'samples/real', orglabel, labels, 'real')


if __name__ == '__main__':
    setup_seed(1024)
    parser = argparse.ArgumentParser(description='Audio_advGAN')
    parser.add_argument('--epochs', type=int, default=60, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')

    parser.add_argument('--g_lr', type=float, default=1e-4, help='')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='')
    parser.add_argument('--train_dataset', type=str, default=r'E:\exp\speech_commands\train', help='datasets/speech_commands/train')
    parser.add_argument('--valid_dataset', type=str, default=r'E:\exp\speech_commands\valid', help='datasets/speech_commands/valid')
    parser.add_argument('--test_dataset', type=str, default=r'E:\exp\speech_commands\test', help='datasets/speech_commands/test')
    parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='')# resnext(13)+denset(16)  resnet18(5)+widerestnet(11)
    parser.add_argument('--model', choices=models.available_models, default=models.available_models[0],
                        help='model of NN')                                     # dpn92(14) + vgg19(4)
    parser.add_argument('--model1', choices=models.available_models, default=models.available_models[0],
                        help='model of NN')
    parser.add_argument('--model2', choices=models.available_models, default=models.available_models[1],
                        help='model of NN')
    parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
    parser.add_argument("--pre_trained", type=bool, default=True, help='checkpoint file to resume')
    parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau',
                        help='method to adjust learning rate')
    parser.add_argument("--lr-scheduler-patience", type=int, default=3,
                        help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument("--lr-scheduler-step-size", type=int, default=50,
                        help='lr scheduler step: number of epochs of learning rate decay.')
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1,
                        help='learning rate is multiplied by the gamma to decrease it')
    parser.add_argument("--max-epochs", type=int, default=100, help='max number of epochs')
    parser.add_argument("--optim", choices=['adam'], default='adam', help='choices of optimization algorithms')
    parser.add_argument("--is_targeted", type=bool, default=True, help='is target ')
    parser.add_argument("--target", type=str, default='yes', help='the target you wanted to attack')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Use GPU', device)
    if use_gpu:
        torch.backends.cudnn.benchmarks = True

    train_dataset = SpeechCommandsDataset_classifier(args.train_dataset,args.target)
    valid_dataset = SpeechCommandsDataset_classifier(args.valid_dataset,args.target)
    test_dataset = SpeechCommandsDataset_classifier(args.test_dataset,args.target)  

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_gpu,
                              num_workers=args.dataload_workers_nums, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, pin_memory=use_gpu,
                              num_workers=args.dataload_workers_nums, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=use_gpu,
                             num_workers=args.dataload_workers_nums, drop_last=True)

    # a name used to save checkpoints etc.
    full_name = '%s_%s_%s_bs%d_new_mfcc' % (
        args.model, args.optim, args.lr_scheduler, args.batch_size)
    if args.comment:
        full_name = '%s_%s' % (full_name, args.comment)
    loss_summar_valid = {'loss_g':[],'loss_gan':[],'loss_adv':[],'loss_hinge':[], 'loss_l2':[],'acc':[]}
    loss_summar_train = {'loss_g':[],'loss_gan':[],'loss_adv':[],'loss_hinge':[], 'loss_l2':[],'acc':[]}
    f = models.create_model(model_name=args.model, num_classes=10, in_channels=1).to(device)
    # f1 = models.create_model(model_name=args.model1, num_classes=10, in_channels=1).to(device)
    # f2 = models.create_model(model_name=args.model2, num_classes=10, in_channels=1).to(device)
    G = CGenerator()
    D = CDiscriminator()
    mfcc_layer = MFCC(winlen=0.02, winstep=0.032, numcep=26).to(device)  # MFCC layer
    if use_gpu:
        f = torch.nn.DataParallel(f).cuda()
        G = torch.nn.DataParallel(G).cuda()
        D = torch.nn.DataParallel(D).cuda()
    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    criterion_crossentropy = nn.CrossEntropyLoss()

    if use_gpu:
        criterion_gan.to(device)
        criterion_l1.to(device)
        criterion_l2.to(device)
        criterion_crossentropy.to(device)

    g_optimizer = optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    start_timestamp = int(time.time() * 1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    global_step = 0

    if args.pre_trained:
        print("Loading a pretrained model ")

        # best-acc-speech-commands-checkpoint-resnext29_8_64.pth
        # best-acc-speech-commands-checkpoint-dpn92.pth
        # best-acc-speech-commands-checkpoint-resnet18.pth
        checkpoint = torch.load(
            os.path.join(args.checkpoint, 'wideresnet28_10_9414.pth'))
        f.load_state_dict(checkpoint['state_dict'])
        # best-acc-speech-commands-checkpoint-densenet_bc_250_24.pth
        # best-acc-speech-commands-checkpoint-vgg19_bn.pth
        # best-acc-speech-commands-checkpoint-wideresnet28_10.pth
        # checkpoint1 = torch.load(
        #     os.path.join(args.checkpoint, 'speechcommand/best-acc-speech-commands-checkpoint-dpn92.pth'))
        # f1.load_state_dict(checkpoint1['state_dict'])
        # checkpoint2 = torch.load(
        #     os.path.join(args.checkpoint, 'speechcommand/best-acc-speech-commands-checkpoint-vgg19_bn.pth'))
        # f2.load_state_dict(checkpoint2['state_dict'])
        #
        # checkpoint = torch.load(os.path.join(args.checkpoint, 'speechcommand/sampleCNN_49.pth'))
        # f.load_state_dict(checkpoint)
        del checkpoint
        # del checkpoint1
        # del checkpoint2
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, patience=args.lr_scheduler_patience,
                                                                  factor=args.lr_scheduler_gamma)
    def get_lr():
        return g_optimizer.param_groups[0]['lr']


    print("training %s for Google speech commands..." % args.model)
    since = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        acc = valid(epoch)
        lr_scheduler.step(acc)
        if epoch % 2 ==0:
            test(epoch)
        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100 * best_accuracy, best_loss))
    np.save('loss_summar_with_hinge_train', loss_summar_train)
    np.save('loss_summar_with_hinge_valid', loss_summar_valid)
    print("finished")