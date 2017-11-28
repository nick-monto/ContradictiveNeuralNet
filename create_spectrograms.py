#!/usr/bin/env python
import os
import math
import numpy as np
import matplotlib.pylab as plt
from timeit import default_timer as timer
from my_spectrogram import my_specgram
from scipy.io import wavfile
from pylab import rcParams


rcParams['figure.figsize'] = 7, 4

SCRIPT_DIR = os.getcwd() + '/'
INPUT_FOLDER = 'Input_audio/'
OUTPUT_FOLDER = 'Input_spectrogram/'

num = int(input('How many spectrograms would you like to create for each audio file? '))

def plot_spectrogram(audiopath, plotpath=None, NFFT_window=0.025,
                     noverlap_window=0.023, freq_min=None, freq_max=None,
                     axis='off'):
    fs, data = wavfile.read(audiopath)
    data = data / data.max()
    center = data.mean() * 0.2
    data = data + np.random.normal(center, abs(center * 0.5), len(data))
    NFFT = pow(2, int(math.log(int(fs*NFFT_window), 2) + 0.5))  # 25ms window, nearest power of 2
    noverlap = int(fs*noverlap_window)
    fc = int(np.sqrt(freq_min*freq_max))
    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance
    Pxx, freqs, bins, im = my_specgram(data, NFFT=NFFT, Fs=fs,
                                       Fc=fc, detrend=None,
                                       window=np.hanning(NFFT),
                                       noverlap=noverlap, cmap='Greys',
                                       xextent=None,
                                       pad_to=None, sides='default',
                                       scale_by_freq=None,
                                       minfreq=freq_min, maxfreq=freq_max)
    plt.axis(axis)
    im.axes.axis('tight')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    if plotpath:
        plt.savefig(plotpath, bbox_inches='tight',
                    transparent=False, pad_inches=0, dpi=96)
    else:
        plt.show()
    plt.clf()


# same as training but no added noise
def plot_spectrogram_val(audiopath, plotpath=None, NFFT_window=0.025,
                         noverlap_window=0.023, freq_min=None, freq_max=None,
                         axis='off'):
    fs, data = wavfile.read(audiopath)
    data = data / data.max()
    NFFT = pow(2, int(math.log(int(fs*NFFT_window), 2) + 0.5))  # 25ms window, nearest power of 2
    noverlap = int(fs*noverlap_window)
    fc = int(np.sqrt(freq_min*freq_max))
    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance
    Pxx, freqs, bins, im = my_specgram(data, NFFT=NFFT, Fs=fs,
                                       Fc=fc, detrend=None,
                                       window=np.hanning(NFFT),
                                       noverlap=noverlap, cmap='Greys',
                                       xextent=None,
                                       pad_to=None, sides='default',
                                       scale_by_freq=None,
                                       minfreq=freq_min, maxfreq=freq_max)
    plt.axis(axis)
    im.axes.axis('tight')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    if plotpath:
        plt.savefig(plotpath, bbox_inches='tight',
                    transparent=False, pad_inches=0, dpi=96)
    else:
        plt.show()
    plt.clf()


training_list = sorted(os.listdir(INPUT_FOLDER + 'Train/'))
if not os.path.exists(OUTPUT_FOLDER + 'Training'):
    os.makedirs(OUTPUT_FOLDER + 'Training')
    print('Successfully created a training folder!')
print('Populating training folder with spectrograms...')
for j in training_list:
    start = timer()
    for k in range(0, num):
        plot_spectrogram(SCRIPT_DIR + INPUT_FOLDER + 'Train/' + j,
                                     plotpath=OUTPUT_FOLDER + 'Training/' +
                                     str(j[:-13]) +
                                     str(k) + '.jpeg',
                                     NFFT_window=0.025, noverlap_window=0.023,
                                     freq_min=0, freq_max=8000)
    end = timer()
    print('Finished converting {} in {} seconds.'.format(j[:-14], end-start))


# validation_list = sorted(os.listdir(INPUT_FOLDER + 'Val/'))
if not os.path.exists(OUTPUT_FOLDER + 'Validation'):
    os.makedirs(OUTPUT_FOLDER + 'Validation')
    print('Successfully created a validation folder!')
print('Populating validation folder with spectrograms...')
for j in training_list:
    start = timer()
    for k in range(0, 1):
        plot_spectrogram_val(SCRIPT_DIR + INPUT_FOLDER + 'Validation/' + j,
                                            plotpath=OUTPUT_FOLDER + 'Validation/' +
                                            str(j[:-13]) +
                                            str(k) + '.jpeg',
                                            NFFT_window=0.025, noverlap_window=0.023,
                                            freq_min=0, freq_max=8000)
    end = timer()
    print('Finished converting {} in {} seconds.'.format(j[:-14], end-start))
#
#
# test_list = sorted(os.listdir(INPUT_FOLDER + 'Test/'))
# if not os.path.exists(OUTPUT_FOLDER + 'Test'):
#     os.makedirs(OUTPUT_FOLDER + 'Test')
#     print('Successfully created a test folder!')
# print('Populating test folder with spectrograms...')
# for j in test_list:
#     start = timer()
#     for k in range(0, 1):
#         plot_spectrogram_val(SCRIPT_DIR + INPUT_FOLDER + 'Test/' + j,
#                                             plotpath=OUTPUT_FOLDER + 'Test/' +
#                                             str(j[:-13]) +
#                                             str(k) + '.jpeg',
#                                             NFFT_window=0.025, noverlap_window=0.023,
#                                             freq_min=0, freq_max=8000)
#     end = timer()
#     print('Finished converting {} in {} seconds.'.format(j[:-14], end-start))
