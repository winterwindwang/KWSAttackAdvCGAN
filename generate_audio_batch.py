import torch
import torchvision
from models.generator import Generator, CGenerator
from scipy.io import wavfile
from datasets import *
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import numpy as np
import time
from torch.autograd import Variable

def one_hot(label, num_class=10):
    '''
    :param batch_size:
    :param label: should be the shape [B,1], torch.squeeze(label,1)
    :param num_class:
    :return:
    '''
    label = torch.unsqueeze(label, dim=1)
    return torch.zeros(label.size(0), num_class).scatter(1, label, 1)

def wav_save(epoch, data, data_dir, label, target, name):
    singals = data.cpu().data.numpy()
    label = label.cpu().data.numpy()
    idx2classes = {0: 'yes', 1: 'no', 2: 'up',3:'down',4:'left',5:'right',6:'on',7:'off',8:'stop',9:'go'}
    for i in range(len(singals)):
        output = singals[i].reshape(16384, 1)
        output = (output - 1) / (2 / 65535) + 32767
        output = output.astype(np.int16)
        labels = idx2classes[label[i]]
        dir = os.path.join(data_dir, labels)
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        filename = "{}_{}_to_{}_epoch_{}_{}.wav".format(name, idx2classes[label[i]], idx2classes[target], epoch, i)
        path = os.path.join(dir, filename)
        wavfile.write(path, 16384, output)

def default_loader(path, sample_rate=16384):
    fn, wav_data = wavfile.read(path)
    if sample_rate < len(wav_data):
        wav_data = wav_data[:sample_rate]
    elif sample_rate > len(wav_data):
        wav_data = np.pad(wav_data, (0, sample_rate - len(wav_data)), "constant")
    wav_data = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
    return wav_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio_advGAN')
    parser.add_argument('--test_dataset', type=str, required=True,help='the test data dir')
    parser.add_argument('--checkpoint', type=str,required=True, help='the checkpoint file')
    parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
    parser.add_argument("--output-dir", type=str, required=True, help='the output generated adversarial examples dir')
    parser.add_argument('--target', type=int, default=0)
    args = parser.parse_args()
    targets = {0: 'yes', 1: 'no', 2: 'up',3:'down',4:'left',5:'right',6:'on',7:'off',8:'stop',9:'go'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = torch.cuda.is_available()

    G = CGenerator()
    if use_gpu:
        G = torch.nn.DataParallel(G).cuda()
    ckpt = torch.load(args.checkpoint)
    G.load_state_dict(ckpt['state_dict'])

    for target in targets.keys():
        print('攻击目标为%s'%targets[target])
        file_dir = args.output_dir + '\攻击目标为%s'% targets[target]
        output_gen = r'\generated\gen'
        output_real = r'\generated\real'
        gen_dir  = file_dir + output_gen
        real_dir = file_dir + output_real
        if os.path.exists(gen_dir) is False:
            os.makedirs(gen_dir)
            os.makedirs(real_dir)
        test_dataset = SpeechCommandsDataset_batch(args.test_dataset, targets[target])  # valid_feature_transform
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
        pbar = tqdm(test_loader, unit='generate audio', unit_scale=test_loader.batch_size)
        idx = 0
        num = 0
        time_sum = []
        LongTensor = torch.LongTensor
        for input, label in pbar:

            start_time = time.time()
            input = torch.unsqueeze(input, 1)

            labels = torch.ones_like(label).fill_(target)
            if use_gpu:
                labels = labels.to(device)
                input = input.to(device)
                label = label.to(device)

            perturbation = torch.clamp(G(input, labels), -0.3, 0.3)
            adv_audio = perturbation + input
            gen = adv_audio.clamp(-1., 1.)

            wav_save(idx, gen, gen_dir, label, target, 'fake')
            wav_save(idx, input, real_dir, label, target, 'real')
            end_time = time.time()
            epoch_time = end_time - start_time
            time_sum.append(epoch_time)
            print("Attack done in %0.4f seconds" % (end_time - start_time))
            idx += 1
            num += input.size(0)
        print('Total examples :', num)
        print('Total time :', sum(time_sum))
        print('Finished!')
        del test_dataset, test_loader

