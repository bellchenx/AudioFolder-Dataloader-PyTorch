import torch.utils.data as data
import torchaudio
import os
import os.path

AUDIO_EXTENSIONS = ['.mp3', '.wav']

def is_audio_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    audio = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    audio.append(item)

    return audio

class AudioFolder(data.Dataset):
    def __init__(self, root, transform=None):
        classes, class_to_idx = find_classes(root)
        audios = make_dataset(root, class_to_idx)
        if len(audios) == 0:
            raise(RuntimeError("Found 0 audios in subfolders of: " + root + "\n"
                               "Supported audio extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.audios = audios
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.audios[index]
        audio, _ = torchaudio.load(path)
        if self.transform is not None:
            audio = self.transform(audio)

        return audio, target

    def __len__(self):
        return len(self.audios)