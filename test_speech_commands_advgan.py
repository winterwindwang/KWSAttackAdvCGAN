import argparse
import os
import models

from tqdm import *
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchnet.meter as meter
from pytorch_mfcc import MFCC

from datasets import *
# from transforms import *
import scipy.io.wavfile as wavfile
from glob import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp


def multi_crop(inputs):
    b = 1
    size = inputs.size(3) - b *2
    patches = [int[:, :, :, i * b:size+i*b] for i in range(3)]
    outputs = torch.stack(patches)
    outputs = outputs.view(-1, inputs.size(1), inputs.size(2), size)
    outputs = torch.nn.functional.pad(outputs, (b,b,0,0), mode='predicate')
    return torch.cat((inputs, outputs.data))


def test():
    f.eval() # set model to evluate mode

    # running_loss = 0
    # it = 0
    correct = 0
    total = 0
    confusion_matrix = meter.confusionmeter.ConfusionMeter(10)
    pbar = tqdm(test_loader, unit='audios', unit_scale=test_loader.batch_size)
    for input, label in pbar:
        input = torch.unsqueeze(input, 1)
        lengths = [input.size(2) for _ in range(input.size(0))]
        val, mfcc_lengths = mfcc_layer(torch.squeeze(input.detach()), lengths)
        inputs = Variable(torch.unsqueeze(val, dim=1), requires_grad=True)

        if use_gpu:
            inputs = inputs.cuda()
            targets = label.cuda(async=True)
        # forward
        outputs = f(inputs)
        pred = outputs.data.max(1, keepdim=True)[1]
        cpu_data = outputs.cpu().data.numpy()
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        confusion_matrix.add(torch.squeeze(pred), targets.data)

        accuacy = correct.item() / total
        print('accuracy: %f%%' %(100 * accuacy))
        print('confusion matrix:')
        print(confusion_matrix.value())



def wav_snr(ref_wav, in_wav): # if ref_wav large than in_wav，then pad 0 to in_wav
    if (abs(in_wav.shape[0] - ref_wav.shape[0]) < 10):
        pad_width = ref_wav.shape[0] - in_wav.shape[0]
        in_wav = np.pad(in_wav, (0, pad_width), 'constant')
    else:
        print("Error：the length of ref_wav and in_wav is different")
        return -1

    # Calculate SNR
    norm_diff = np.square(np.linalg.norm(in_wav - ref_wav))
    if (norm_diff == 0.):
        print("Error：ref_wav and in_wav are the same file")
        return -1

    ref_norm = np.square(np.linalg.norm(ref_wav))
    snr = 10 * np.log10(ref_norm / norm_diff)
    return snr


def processor(real_file, fake_file, folder):
    '''
    calculate the file .wav snr
    :param real_file:
    :param fake_file:
    :return:
    '''
    real_list = glob(os.path.join(real_file,'*.wav'))
    fake_list = glob(os.path.join(fake_file,'*.wav'))
    snr = []
    idx = 0
    for (i, j) in zip(real_list, fake_list):
        fake_signal = wavfile.read(i)[1]
        real_signal = wavfile.read(j)[1]
        res = wav_snr(fake_signal, real_signal)   # SNR
        # res = pesq(16000, real_signal, fake_signal,'wb')     # PESQ
        snr.append(res)
        idx += 1
    return np.sum(snr) / len(snr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=32, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=3, help='number of workers for dataloader')
    parser.add_argument('--multi-crop', action='store_true', help='apply crop and average the results')
    parser.add_argument('--generate-kaggle-submission', action='store_true', help='generate kaggle submission file')
    parser.add_argument("--test-dataset-dir", type=str, required=True,
                        help='path of kaggle test dataset')
    parser.add_argument('--output', type=str, default='',
                        help='save output to file for the kaggle competition, if empty the model name will be used')
    parser.add_argument('--model', choices=models.available_models, default=models.available_models[16],
                        help='model of NN')
    parser.add_argument('--checkpoint', type=str, required=True, help='the folder of checkpoint file')
    parser.add_argument("--dataset-dir", type=str, default='generated/gen', help='path of generated data')
    parser.add_argument('--save-dir', type=str, default='data')
    args = parser.parse_args()

    model_name = 'wideresnet'
    target = 'yes'
    dataset_dir = args.test_dataset_dir+'\攻击目标为%s\generated\gen' % target
    dir_file_real = args.test_dataset_dir+'\攻击目标为%s\generated\real' % target
    dir_file_gen = args.test_dataset_dir+'\攻击目标为%s\generated\gen' % target
    print('loading model...')
    if model_name == 'densenet_bc_250_24':
        f = models.create_model(model_name=models.available_models[16], num_classes=10, in_channels=1)
        checkpoint = torch.load(args.checkpoint, 'densenet_bc_250_24.pth')
        test_dataset = SpeechCommandsDataset_batch(dataset_dir, target)
    elif model_name == 'vgg19_bn':
        f = models.create_model(model_name=models.available_models[4], num_classes=10, in_channels=1)
        checkpoint = torch.load(args.checkpoint, 'vgg19_bn.pth')
        test_dataset = SpeechCommandsDataset_batch(dataset_dir, target)
    elif model_name == 'resnet18':
        f = models.create_model(model_name=models.available_models[5], num_classes=10, in_channels=1)
        checkpoint = torch.load(args.checkpoint, 'resnet18.pth')
        test_dataset = SpeechCommandsDataset_batch(dataset_dir, target)
    elif model_name == 'resnext29_8_64':
        f = models.create_model(model_name=models.available_models[13], num_classes=10, in_channels=1)
        checkpoint = torch.load(args.checkpoint, 'resnext29_8_64.pth')
        test_dataset = SpeechCommandsDataset_batch(dataset_dir, target)
    elif model_name == 'dpn92':
        f = models.create_model(model_name=models.available_models[14], num_classes=10, in_channels=1)
        checkpoint = torch.load(args.checkpoint, 'dpn92.pth')
        test_dataset = SpeechCommandsDataset_batch(dataset_dir, target)
    elif model_name == 'wideresnet':
        f = models.create_model(model_name=models.available_models[10], num_classes=10, in_channels=1)
        checkpoint = torch.load(args.checkpoint, 'wideresnet28_10_9414.pth')
        test_dataset = SpeechCommandsDataset_batch(dataset_dir, target)
    else:
        print("There is non specific model")
        raise FileNotFoundError
    f.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    # f.eval()
    # print(f)
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        f = torch.nn.DataParallel(f).cuda()
    if isinstance(f, torch.nn.DataParallel):
        f = f.module
    f.eval()
    n_mels = 32
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_gpu,
                                  num_workers=args.dataload_workers_nums)
    pbar = tqdm(test_loader, unit='generate audio', unit_scale=test_loader.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mfcc_layer = MFCC(winlen=0.02, winstep=0.032, numcep=26).to(device)  # MFCC layer

    criterion = torch.nn.CrossEntropyLoss()
    test()
    # SNR Calculation
    snr_sum = 0
    all_classes = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')  #
    # all_classes = 'blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock'.split(',')
    for i in range(len(all_classes)):
        if all_classes[i] == target:
            continue
        snr = processor(os.path.join(dir_file_real, all_classes[i]), os.path.join(dir_file_gen, all_classes[i]),
                        all_classes[i])
        print('snr for the class {} is {}'.format(all_classes[i], snr))
        snr_sum += snr
    print(snr_sum / 9)


