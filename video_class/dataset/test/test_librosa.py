'''
@Author: your name
@Date: 2022-01-11 20:38:06
@LastEditTime: 2022-01-12 16:47:19
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /gaoqingdong/图短小融合/model/UVAT/test2/UVAT/dataset/test/test_librosa.py
'''
import torch
import librosa
from scipy.io import wavfile
import os
root_path = ''
def get_audio_input(self, info):
    if 'video_audio_path' not in info:
        return torch.zeros(0, dtype=torch.float), 0
    sr, wav = wavfile.read(os.path.join(root_path, info['video_audio_path']))
    wav = wav/max(wav)
    print(len(wav))
    #pritn(len(wav)/ (sr/))
    audio_mfcc = librosa.feature.mfcc(wav, sr, n_mfcc = 13)
    audio_mfcc = torch.tensor(librosa.feature.mfcc(wav, sr, n_mfcc = 13))
    #len_audio = (min(config.max_audio, len(wav))) // 512 * 512
    #wav = np.array(wav)
    return audio_mfcc#torch.tensor(wav[:len_audio]), len_audio

if __name__=="__main__":
    
    info = {"video_audio_path": "/home/bpfsrw_8/gaoqingdong/抽帧/audio/3886937018027595418.mp4.wav"}
    res= get_audio_input(1,info)
    print(res)