# Module for calculating the metrics for evaluation purposes


# Importing libraries
import os
import oct2py
from pesq import pesq
from scipy.io import wavfile
from tqdm import tqdm
import semetrics
import soundfile as sf
from pydub import AudioSegment

# Defining paths to relevant folders
clean_data = 'data/clean_testset_wav'
enhanced_data = 'data/enhanced-audio'

# Trimming clean audio data to match length of generated audio 

if not os.path.exists(clean_data):
        os.makedirs(clean_data)

for root, dirs, files in os.walk(clean_data):
    if len(files) == 0:
        continue
    for filename in tqdm(files, desc='Trimming clean test audios to match generated audios'):
        f = sf.SoundFile(os.path.join(enhanced_data,"enhanced_"+filename))
        seconds = len(f)/f.samplerate
        myaudio = AudioSegment.from_file(os.path.join(clean_data,filename) , "wav")
        extract = myaudio[:((seconds*1000))]
        extract.export(os.path.join(clean_data,filename), format="wav")

# Initializing metrics
pesq_score = 0
csig = 0
cbak = 0
covl = 0
count=1

# Calculating of metrics using Semetrics module
for root, dirs, files in os.walk(clean_data):
    if len(files) == 0:
        continue
    for filename in tqdm(files, desc='Calculating Metrics for enhanced files'):
        clean_file = os.path.join(clean_data, filename)
        enhanced_file = os.path.join(enhanced_data, "enhanced_" + filename)
        try:

            pesq, cs, cb, co, ss = semetrics.composite(clean_file, enhanced_file)
            pesq_score += pesq
            csig += cs
            cbak += cb
            covl += co
            count=count+1
        except Exception:
            pass
    
    print("Overall PESQ = ", pesq_score/count)
    print("Overall CSIG = ", csig/count)
    print("Overall CBAK = ", cbak/count)
    print("Overall COVL = ", covl/count)


