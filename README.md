
# Speech enhancement using Generative Adversarial Networks

This project utilizes the capabilities of Generative Adversarial Networks to generate enhanced speech from noisy audio files.


## Installation

All files must be executed via the Command Prompt.

How to download and run the code:

Step 1: Download the code either by cloning the repository or using the download option

Step 2: Download the datasets from https://datashare.ed.ac.uk/handle/10283/2791. Download only the following data:
 clean_testset_wav.zip (147.1Mb)
 clean_trainset_28spk_wav.zip (2.315Gb)
 noisy_testset_wav.zip (162.6Mb)
 noisy_trainset_28spk_wav.zip (2.635Gb)

Create a subfolder called 'data' in the same folder where the code files are stored. Unzip the datasets into this folder.

Step 3: Install the following libraries using pip:

```bash
  pip install librosa
  pip install pytorch
  pip install numpy
  pip install scipy
  pip install tqdm
  pip install soundfile
  pip install torchgan
  pip install semetrics
  pip install pydub
  pip install pesq
  pip install oct2py
  pip install tensorboard
```

Step 4: Once done, run the data_preprocess.py file to perform the preprocessing

```bash
  python data_preprocess.py
```

Step 5: Run the main.py file to train the model. All hyperparameters of loss functions can be tweaked in this file. By default the training happens with a batch_size of 200 and epoch step set as 20. However, this can be tweaked by supplying the arguments:

```bash
  python main.py --batch_size 250 --num_epochs 30
```

Step 6: Run the test_audio.py file to generate the enhanced audio files. Supply the epoch name as argument:

```bash
  python test_audio.py -- epoch_name=generator-15.pkl
```

Step 7: Run the evaluation file to generate the metrics for the enhanced audios


```bash
  python evaluation.py
```

Note: setup.py shall be ran to install semetrics to generate the metrics.

```bash
  python setup.py
```

Samples of two audio files are provided for reference.

Tensorboard results can be plotted using the following command:

```bash
  tensorboard dev upload --logdir runs
```

Create a directory named 'run' in the same folder where all your files are stored. This directory will be used to store the tensorboard results
