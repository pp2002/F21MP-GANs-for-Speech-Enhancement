# Module for running the selected model on the supplied test set

# Importing libraries
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
from tqdm import tqdm

from data_preprocess import slice_signal, window_size, sample_rate
from new_model import Generator
from utils import pre_emphasis, de_emphasis
import soundfile as sf

# Defining paths for the relevant folders
noisy_test_folder = 'data/noisy_testset_wav'
enhanced_audio_folder = 'data/enhanced-audio'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('--epoch_name', type=str, required=True, help='generator epoch name') # Epoch name to be entered as argument

    opt = parser.parse_args()
    EPOCH_NAME = opt.epoch_name

    generator = Generator()
    generator.load_state_dict(torch.load('epochs/' + EPOCH_NAME, map_location='cpu')) # Loading pickle file of generator
    if torch.cuda.is_available():
        generator.cuda()
    
    if not os.path.exists(enhanced_audio_folder):
        os.makedirs(enhanced_audio_folder)

    # Generating enhanced speech
    for root, dirs, files in os.walk(noisy_test_folder):
        for filename in tqdm(files, desc='Generate enhanced audios'):
            noisy_file = os.path.join(noisy_test_folder, filename)
            noisy_slices = slice_signal(noisy_file, window_size, 1, sample_rate)
            enhanced_speech = []
            for noisy_slice in noisy_slices:
                z = nn.init.normal_(torch.Tensor(1, 1024, 8))
                noisy_slice = torch.from_numpy(pre_emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
                if torch.cuda.is_available():
                    noisy_slice, z = noisy_slice.cuda(), z.cuda()
                noisy_slice, z = Variable(noisy_slice), Variable(z)
                generated_speech = generator(noisy_slice, z).data.cpu().numpy()
                generated_speech = de_emphasis(generated_speech, emph_coeff=0.95)
                generated_speech = generated_speech.reshape(-1)
                enhanced_speech.append(generated_speech)

            enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
            file_name = os.path.join(enhanced_audio_folder,
                             'enhanced_{}.wav'.format(os.path.basename(noisy_file).split('.')[0]))
            sf.write(file_name, enhanced_speech.T, sample_rate, subtype='PCM_16') # Saving enhanced speech with same sample rate and bit rate as that of the original file





