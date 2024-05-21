import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import torch
import librosa
import os
import os.path

def get_files(paths, split='train'):
    files = []
    for path in paths:
        for emotion in os.listdir(os.path.join(path, split)):
            for basename in os.listdir(os.path.join(path, split, emotion)):
                files.append(os.path.join(path, split, emotion, basename))
    return files


def load_data(paths, split='train'):
    # Retrieve all files in chosen path with the specific extension 
    audios_path = get_files(paths, split)
    # Get the file name as its label
    labels = [path.split('/')[-2] for path in audios_path]
    if len(audios_path) == 0:
        print('There is no sample in dataset')
        quit()
    else:
        # Encode labels to digits
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        df = pd.DataFrame({"path": audios_path, "label": labels})
        return df, le


class SpeechDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tsr = None

    def __getitem__(self, idx):
        path = self.data.path.values[idx]
        label = self.data.label.values[idx]
        speech = self.file_to_array(path, self.tsr)
        return speech, label

    def file_to_array(self, path, sampling_rate):
        array, _ = librosa.load(path, sr=sampling_rate)
        return array

    def __len__(self):
        return len(self.data)


def collate_fn_padd(batch):
    batch = np.array(batch, dtype=object)
    speeches = batch[:, 0]
    labels = batch[:, 1]

    max_len = 144000
    speeches = np.vstack(
        [
            np.pad(speech, (0, max_len - len(speech)), 'constant', constant_values=0) 
            if max_len > len(speech) else speech[: max_len] for speech in speeches
        ]
    )
    speeches = torch.from_numpy(speeches).float()

    labels = np.vstack(labels).astype(float)
    labels = torch.from_numpy(labels).squeeze().type(torch.LongTensor)

    return speeches, labels


def get_data_loaders(train_data, train_bs):
    train_dl = DataLoader(SpeechDataset(train_data), batch_size=train_bs, collate_fn=collate_fn_padd, shuffle=True)
    return train_dl
