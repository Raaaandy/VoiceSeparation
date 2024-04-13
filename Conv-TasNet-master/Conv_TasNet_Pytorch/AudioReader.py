import glob
import os

import torchaudio
import torch
from utils import handle_scp


def read_wav(fname, return_rate=False):
    """
         Read wavfile using Pytorch audio
         input:
               fname: wav file path
               return_rate: Whether to return the sampling rate
         output:
                src: output tensor of size C x L
                     L is the number of audio frames
                     C is the number of channels.
                sr: sample rate
    """
    src, sr = torchaudio.load(fname, channels_first=True)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()


def write_wav(fname, src, sample_rate):
    '''
         Write wav file
         input:
               fname: wav file path
               src: frames of audio
               sample_rate: An integer which is the sample rate of the audio
         output:
               None
    '''
    torchaudio.save(fname, src, sample_rate)


class AudioReader(object):
    '''
        Class that reads Wav format files
        Input as a different scp file address
        Output a matrix of wav files in all scp files.
    '''

    def __init__(self, scp_path, sample_rate=8000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.path = scp_path
        # pattern = os.path.join(scp_path, '**', '*.flac')
        self.index_dict = {}
        for file in os.listdir(scp_path):
            if file in self.index_dict:
                raise RuntimeError("This path has been in the dict")
            self.index_dict[file] = os.path.join(scp_path, file)
        # 使用glob.glob搜索所有匹配的文件路径，包括所有子目录

        self.keys = list(self.index_dict.keys())

    def _load(self, key):
        src, sr = read_wav(self.index_dict[key], return_rate=True)
        if self.sample_rate is not None and sr != self.sample_rate:
            raise RuntimeError('SampleRate mismatch: {:d} vs {:d}'.format(
                sr, self.sample_rate))
        return src

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for key in self.keys:
            yield key, self._load(key)

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError('Unsupported index type: {}'.format(type(index)))
        if type(index) == int:
            num_uttrs = len(self.keys)
            if num_uttrs < index and index < 0:
                raise KeyError('Interger index out of range, {:d} vs {:d}'.format(
                    index, num_uttrs))
            print(self.index_dict.keys())
        if index not in self.index_dict:
            print(index)
            raise KeyError("Missing utterance {}!".format(index))

        return self._load(index)


if __name__ == "__main__":
    r = AudioReader('C:\\Users\\Randy\\Downloads\\Conv-TasNet-master\\Conv-TasNet-master\\Conv_TasNet_Pytorch\\chunk1')
    print(r.keys)
    index = 0
    print(r[1])
    print(len(r))
