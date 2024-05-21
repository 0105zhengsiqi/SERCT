from svm import SVM
from dataset import load_data, get_data_loaders
import torch
import numpy as np
import random
from typing import List


def trainModel(
    data_paths: List[str], 
    train_bs: int, 
):

    # load data
    df_train, label_encoder = load_data(data_paths, split='train')
    df_train = df_train.sample(frac=1, random_state=42)
    num_classes = len(label_encoder.classes_)
    print(f'num_classes: {num_classes}')

    df_test, _ = load_data(data_paths, split='test')
    dataset_name = data_paths[0].split('/')[-1]
    print('Data loaded successfully')

    # Create data loaders
    train_dataloader = get_data_loaders(df_train, train_bs)
    test_dataloader = get_data_loaders(df_test, train_bs)
    x_train, y_train = [], []
    for (inputs, labels) in train_dataloader:
        for i, l in zip(inputs, labels):
            x_train.append(i.numpy())
            y_train.append(l.item())

    x_train = np.vstack(x_train)
    y_train = np.array(y_train)

    x_test, y_test = [], []
    for (inputs, labels) in test_dataloader:
        for i, l in zip(inputs, labels):
            x_test.append(i.numpy())
            y_test.append(l.item())

    x_test = np.vstack(x_test)
    y_test = np.array(y_test)
    
    params = {
        "kernel": 'rbf',
        "probability": True,
        "gamma": 'auto'
    }
    model = SVM.make(params)
    print('Model loaded successfully')

    print('Start Training ....')
    model.train(x_train, y_train)
    model.save('ckpts/svm/', f'svm-{dataset_name}-last')
    test_wa, test_ua, test_wf1 = model.evaluate(x_test, y_test, num_classes)

    with open("exps/svm/log.txt", "a") as f:
        f.write(
            f'Test WA = {test_wa:.4f} / Test UA = {test_ua:.4f} / Test WF1 = {test_wf1:.4f}\n'
        )


if __name__ == '__main__':
    random_seed = 3
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    DATASET_PATHS = [
        'datasets/ESD',
        # 'datasets/TESS',
    ]
    BATCH_SIZE = 16

    trainModel(DATASET_PATHS, BATCH_SIZE)
