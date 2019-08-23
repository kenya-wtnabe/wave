from scikits.audiolab import wavread
from scikits.talkbox.features import mfcc

audio, fs, enc = wavread('sample.wav')
ceps, mspec, spec = mfcc(audio, nwin=256, nfft=512, fs=fs, nceps=13)
