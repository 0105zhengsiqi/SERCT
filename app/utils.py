import os
from typing import List
from model import CNNTransformer
import torch
import librosa
import numpy as np

def get_wavs() -> List[str]:
    basenames = os.listdir('emotions')
    return basenames

idx_to_emotion = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

def collate_fn_padd(speech):
    max_len = 144000
    speech = (
        np.pad(speech, (0, max_len - len(speech)), 'constant', constant_values=0)
        if max_len > len(speech) else speech[: max_len]
    )
    speech = speech.reshape((1, -1))
    speech = torch.from_numpy(speech).float()
    return speech

def file_to_array(path, sampling_rate=None):
        array, _ = librosa.load(path, sr=sampling_rate)
        array = collate_fn_padd(array)
        return array

def build_model():
    model_args = {
        "cnn_mode": 'default',
        'conv_layers': [(512, 10, 5, 0)] + [(512, 3, 2, 0)] * 4 + [(512, 2, 2, 0)] + [(512, 2, 2, 1)],
        # (dim, kernel_size, stride, padding)
        'cnn_dropout': 0.0,
        'conv_bias': False,
        'conv_type': "default",
        'input_dim': 1024,
        'length': 450,
        'ffn_embed_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'num_classes': 7,
        'trans_dropout': 0.1,
        'bias': True,
        'activation': 'relu'
    }
    model = CNNTransformer(**model_args)
    device = torch.device('cpu')
    ckpt = torch.load('ckpt/cnn-transformer-mix-50.pt', map_location=device)
    model.load_state_dict(ckpt.state_dict())
    model = model.to(device)
    return model

def infer(path, model, device=torch.device('cpu')):
    speech = file_to_array(path)
    # put the model on evaluation mode
    model.eval()

    inputs = speech.to(device)
    outputs = model(inputs)
    pred = torch.argmax(outputs, dim=1).cpu().numpy()
    emotion = idx_to_emotion[pred[0]]

    return emotion
