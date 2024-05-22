from model import CNNTransformer, load_data, get_data_loaders
import torch
import numpy as np
import random
from utils import compute_unweighted_accuracy, compute_weighted_f1
from typing import List, Dict
from torch import nn
from tqdm import tqdm
import argparse


def collect(outputs, labels, predictions, true_labels):
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    if len(predictions) == 0:
        predictions = preds
        true_labels = labels
    else:
        predictions = np.concatenate((predictions, preds))
        true_labels = np.concatenate((true_labels, labels))

    return predictions, true_labels


def evaluate(model, dataloader, device, num_classes):
    # put the model on evaluation mode
    model.eval()
    predictions, true_labels = [], []

    for (inputs, labels) in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        # Collect predictions and true labels
        predictions, true_labels = collect(outputs, labels, predictions, true_labels)

    # unweighted accuracy
    unweightet_correct = [0] * num_classes
    unweightet_total = [0] * num_classes

    # weighted f1
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    total = len(true_labels)
    correct = (predictions == true_labels).sum()

    for i in range(len(true_labels)):
        unweightet_total[true_labels[i]] += 1
        if predictions[i] == true_labels[i]:
            unweightet_correct[true_labels[i]] += 1
            tp[true_labels[i]] += 1
        else:
            fp[predictions[i]] += 1
            fn[true_labels[i]] += 1

    weighted_acc = correct / total
    unweighted_acc = compute_unweighted_accuracy(unweightet_correct, unweightet_total)
    weighted_f1 = compute_weighted_f1(tp, fp, fn, unweightet_total)

    return (
        weighted_acc,
        unweighted_acc,
        weighted_f1,
        predictions,
        true_labels
    )


def evalModel(
    data_paths: List[str], 
    train_bs: int, 
    model_args: Dict,
    ckpt_path: str,
):

    # load data
    df_test, label_encoder = load_data(data_paths, split='test')
    dataset_name = data_paths[0].split('/')[-1]
    num_classes = len(label_encoder.classes_)
    # model_args['num_classes'] = num_classes
    num_classes = model_args['num_classes']
    print(f'num_classes: {num_classes}')
    print('Data loaded successfully')

    # Create data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(device, 'is available; number of GPUs =', num_gpus)
    test_dataloader = get_data_loaders(df_test, train_bs * num_gpus)
    print('Number of eval batches =', len(test_dataloader))

    print('Loading model ...')
    model = CNNTransformer(**model_args)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt.state_dict())
    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    print('Model loaded successfully')

    print('Start Eval ....')
    with open("training_log.txt", "a") as f:
        f.write(
            f'\ncnn-transformer-layers4 {dataset_name}\n'
        )
    test_wa, test_ua, test_wf1, _ , _ = evaluate(model, test_dataloader, device, num_classes)
    tqdm.write(
        f'Test WA = {test_wa:.4f} / Test UA = {test_ua:.4f} / Test WF1 = {test_wf1:.4f}'
    )
    with open("training_log.txt", "a") as f:
        f.write(
            f'Test WA = {test_wa:.4f} / Test UA = {test_ua:.4f} / Test WF1 = {test_wf1:.4f}\n'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_layers', type=int)
    args = parser.parse_args()

    random_seed = 3
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    DATASET_PATHS = [f'datasets/{args.dataset_name}']
    BATCH_SIZE = args.batch_size
    CKPT_PATH = args.ckpt_path

    model_args = {
        "cnn_mode": 'default',
        'conv_layers': [(512, 10, 5, 0)] + [(512, 3, 2, 0)] * 4 + [(512, 2, 2, 0)] + [(512, 2, 2, 1)],  # (dim, kernel_size, stride, padding)
        'cnn_dropout': 0.0,
        'conv_bias': False,
        'conv_type': "default",
        'input_dim': 1024,
        'length': 450,
        'ffn_embed_dim': 512, 
        'num_layers': args.num_layers, 
        'num_heads': 8, 
        'num_classes': 7, 
        'trans_dropout': 0.1,
        'bias': True,
        'activation': 'relu'
    }

    evalModel(DATASET_PATHS, BATCH_SIZE, model_args, CKPT_PATH)
