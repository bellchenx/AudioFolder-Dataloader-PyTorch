import torchaudio
import os

sample_rate = 44100
out_sample_rate = 44100
channels = 2
step = sample_rate * 1
length = sample_rate * 3
data_dir = './data'
out_dir = './out'
if not os.path.isdir('out'):
    os.mkdir('out')

idx = 0
for filename in os.listdir(data_dir):
    data, rate = torchaudio.load(os.path.join(data_dir, filename))
    total, channel = data.size()
    if rate == sample_rate and channel == channels:
        idj = 0
        for i in range(0, total, step):
            wavname = os.path.join(out_dir,'music%d-%d.wav'%(idx, idj))
            mp3name = os.path.join(out_dir,'music%d-%d.mp3'%(idx, idj))
            idj += 1
            torchaudio.save(wavname, data[i:i+length], out_sample_rate)
            command1 = 'ffmpeg -i %s -acodec libmp3lame %s -loglevel quiet' %(wavname, mp3name)
            command2 = 'rm %s' %(wavname)
            os.system(command1)
            os.system(command2)
    print('Music %d completed.'%idx)
    idx += 1
