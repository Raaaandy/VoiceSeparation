import glob
import itertools
import os
import random

import librosa
import soundfile as sf
import numpy as np
import numpy.random
import torch
import torchaudio
from AudioReader import read_wav, AudioReader, write_wav


def sum_wav(fname1, fname2):
    src1, sr1 = read_wav(fname=fname1, return_rate=True)
    src2, sr2 = read_wav(fname=fname2, return_rate=True)
    print(src1, sr1)
    print(src2, sr2)
    return src1, src2


def select_sound(path, speakers_id: []) -> object:

    selected_speaker = random.sample(speakers_id, 2)
    path1 = os.path.join(path, selected_speaker[0])
    path2 = os.path.join(path, selected_speaker[1])
    sound1 = random.sample(glob.glob(os.path.join(path1,'**/*.flac'), recursive=True), 1)
    sound2 = random.sample(glob.glob(os.path.join(path2, '**/*.flac'),recursive=True), 1)
    return sound1[0], sound2[0]

def valid_part(sound1, sound2):
    voice1, _ = librosa.core.load(sound1, sr=48000, mono=True)
    voice2, _ = librosa.core.load(sound2, sr=48000, mono=True)
    voice1_trim, begin_end1 = librosa.effects.trim(voice1, top_db=18)
    voice2_trim, begin_end2 = librosa.effects.trim(voice2, top_db=18)
    return voice1_trim, begin_end1, voice2_trim, begin_end2

if __name__ == "__main__":
    files_lists = []
    libpath = "C:\\Users\\Randy\\Downloads\\Conv-TasNet-master\\Conv-TasNet-master\\Conv_TasNet_Pytorch\\LibriSpeech\\dev-clean"
    for i in range(1,1001):
        speakers_id = os.listdir(libpath)
        sound1, sound2 = select_sound(libpath, speakers_id)
        voice1_trim, begin_end1, voice2_trim, begin_end2 = valid_part(sound1, sound2)
        while ((voice1_trim.std() <= 2e-4 or begin_end1[1] - begin_end1[0] < 2000 or
               len(voice2_trim) < 3 * 48000 or len(voice1_trim) < 3 * 48000) or
               voice2_trim.std() <= 2e-4 or begin_end2[1] - begin_end2[0] < 2000) :
            sound1, sound2 = select_sound(libpath, speakers_id)
            voice1_trim, begin_end1, voice2_trim, begin_end2 = valid_part(sound1, sound2)
            print(len(voice2_trim), len(voice1_trim))

        chunk1 = voice1_trim[0:3 * 48000]
        chunk1_8k = librosa.resample(chunk1, orig_sr=48000, target_sr=8000)

        sf.write(f'chunk1\\chunk{i}.wav', chunk1_8k, 8000)

        chunk2 = voice2_trim[0:3 * 48000]
        chunk2_8k = librosa.resample(chunk2, orig_sr=48000, target_sr=8000)
        sf.write(f'chunk2\\chunk{i}.wav', chunk2_8k, 8000)
        combine = chunk1 + chunk2
        itr = 0
        for point1, point2, combine_pt in zip(chunk1, chunk2, combine):
            if point1 + point2 == combine_pt:
                itr += 1
                continue
            else:
                print(itr)
                break
        audio = librosa.resample(combine, orig_sr=48000, target_sr=8000)
        audio /= np.abs(audio).max()
        sf.write(f'mix\\chunk{i}.wav', audio, 8000)

        print(f'audio: {audio}')
