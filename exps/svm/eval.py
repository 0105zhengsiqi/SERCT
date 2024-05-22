from svm import SVM
from dataset import load_data, get_data_loaders
import torch
import numpy as np
import random
from typing import List


def evalModel(
    data_paths: List[str], 
    train_bs: int, 
    ckpt_path: str,
):

    # load data
    df_test, label_encoder = load_data(data_paths, split='test')
    num_classes = len(label_encoder.classes_)
    print(f'num_classes: {num_classes}')
    dataset_name = data_paths[0].split('/')[-1]
    print('Data loaded successfully')

    # Create data loaders
    test_dataloader = get_data_loaders(df_test, train_bs)
    x_test, y_test = [], []
    for (inputs, labels) in test_dataloader:
        for i, l in zip(inputs, labels):
            x_test.append(i.numpy())
            y_test.append(l.item())

    x_test = np.vstack(x_test)
    y_test = np.array(y_test)
    
    model = SVM.load(ckpt_path)
    print('Model loaded successfully')

    print('Start Eval ....')
    test_wa, test_ua, test_wf1 = model.evaluate(x_test, y_test, num_classes)

    with open("exps/svm/log.txt", "a") as f:
        f.write(
            f'\n{dataset_name}\n'
        )
        f.write(
            f'Test WA = {test_wa:.4f} / Test UA = {test_ua:.4f} / Test WF1 = {test_wf1:.4f}\n'
        )


if __name__ == '__main__':
    random_seed = 3
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    DATASET_PATHS = [
        'datasets/EMO-DB_de', # 6 classes
        # 'datasets/ESD_zh', # 5 classes
    ]
    BATCH_SIZE = 16
    CKPT_PATH = 'ckpts/svm/svm-ESD-last.m'

    evalModel(DATASET_PATHS, BATCH_SIZE, CKPT_PATH)
