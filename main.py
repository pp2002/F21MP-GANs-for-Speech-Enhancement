# Some parts of this code are referenced from https://github.com/dansuh17/segan-pytorch 

import argparse
import os

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data_preprocess import *
from new_model import Generator, Discriminator
from utils import AudioDataset, pre_emphasis, de_emphasis
import math

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=200, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='train epochs number')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    torch.backends.cudnn.benchmark = True

    # load data
    print('Loading data...')
    train_dataset = AudioDataset(data_type='train')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)
    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
    # optimizers
    g_optimizer = optim.AdamW(generator.parameters(), lr=0.0001) #TTUR 
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=0.0003) #TTUR
    writer = SummaryWriter(comment="AdamW_TTUR_LR_Novel_GAN") # Tensorboard logging

    for epoch in range(NUM_EPOCHS):

        train_bar = tqdm(train_data_loader)
        for train_batch, train_clean, train_noisy in train_bar:

            # latent vector - normal distribution
            z = nn.init.normal_(torch.Tensor(train_batch.size(0), 1024, 8))
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1
            

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(train_noisy, z)
            gen_outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(gen_outputs ** 2)  # L2 loss - we want them all to be 0
            

            d_loss = 0.5 * (clean_loss + noisy_loss)
            

            discriminator.zero_grad()
            d_loss.backward()
            d_optimizer.step()  # update parameters

            # TRAIN G so that D recognizes G(z) as real
            #generator.zero_grad()
            z = nn.init.normal_(torch.Tensor(train_batch.size(0), 1024, 8))
            if torch.cuda.is_available():
                z = z.cuda()
            z = Variable(z)
            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss

            # backprop + optimize
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            train_bar.set_description(
                'Epoch {}: d__loss {:.4f}, g_loss {:.4f}'
                    .format(epoch + 1, d_loss.data, g_loss.data))

            writer.add_scalar("Discriminator Loss/train", d_loss.data, epoch+1)
            writer.add_scalar("Generator Loss/train", g_loss.data, epoch+1)
            

        # save the model parameters for each epoch
        g_path = os.path.join(os.getcwd(),'epochs/', 'generator-{}.pkl'.format(epoch + 1))
        d_path = os.path.join(os.getcwd(),'epochs/', 'discriminator-{}.pkl'.format(epoch + 1))
        torch.save(generator.state_dict(), g_path)
        torch.save(discriminator.state_dict(), d_path)
    writer.flush()
