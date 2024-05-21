from model import CNNTransformer, load_data, get_data_loaders
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
from utils import plot_training, data_distribution, compute_unweighted_accuracy, compute_weighted_f1
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


def train(model, dataloader, optimizer, criterion, epoch, device):
    # put the model on train mode
    model.train()
    losses, predictions, true_labels = [], [], []

    for iter, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect predictions and true labels
        predictions, true_labels = collect(outputs, labels, predictions, true_labels)

    return np.mean(losses), accuracy_score(true_labels, predictions), predictions, true_labels


def evaluate(model, dataloader, criterion, device, num_classes):
    # put the model on evaluation mode
    model.eval()
    losses, predictions, true_labels = [], [], []

    for (inputs, labels) in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

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
        np.mean(losses),
        weighted_acc,
        unweighted_acc,
        weighted_f1,
        predictions,
        true_labels
    )


def trainModel(
    data_paths: List[str], 
    lr: float, 
    epochs: int, 
    weight_decay: float,
    train_bs: int, 
    save_epoch: int,
    model_args: Dict,
    plot_data_dist: bool = False
):

    # load data
    df_train, label_encoder = load_data(data_paths, split='train')
    df_train = df_train.sample(frac=1, random_state=42)
    num_classes = len(label_encoder.classes_)
    model_args['num_classes'] = num_classes
    print(f'num_classes: {num_classes}')

    df_test, _ = load_data(data_paths, split='test')
    dataset_name = data_paths[0].split('/')[-1]
    print('Data loaded successfully')

    # Plot data distribution
    if plot_data_dist: 
        data_distribution(df_train, label_encoder.classes_)

    # Create data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(device, 'is available; number of GPUs =', num_gpus)
    train_dataloader = get_data_loaders(df_train, train_bs * num_gpus)
    test_dataloader = get_data_loaders(df_test, train_bs * num_gpus)
    print('Number of train batches =', len(train_dataloader))

    print('Loading model ...')
    model = CNNTransformer(**model_args)
    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    print('Model loaded successfully')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    print('Start Training ....')
    loss_list, acc_list = [], []
    for epoch in tqdm(range(epochs)):
        train_loss, _ , _ , _ = train(model, train_dataloader, optimizer, criterion, epoch, device)
        test_loss , test_wa, test_ua, test_wf1, _ , _ = evaluate(model, test_dataloader, criterion, device, num_classes)
        scheduler.step()
        loss_list.append([train_loss, test_loss])
        acc_list.append(test_wa)
        tqdm.write(
            f'Train Loss = {train_loss:.4f} / Test Loss = {test_loss:.4f} / Test WA = {test_wa:.4f} / Test UA = {test_ua:.4f} / Test WF1 = {test_wf1:.4f}'
        )
        with open("training_log.txt", "a") as f:
            f.write(
                f'Train Loss = {train_loss:.4f} / Test Loss = {test_loss:.4f} / Test WA = {test_wa:.4f} / Test UA = {test_ua:.4f} / Test WF1 = {test_wf1:.4f}\n'
            )

        if (epoch + 1) % save_epoch == 0:
            torch.save(model, f'ckpts/cnn-trans/cnn-transformer-{dataset_name}-{epoch + 1}.pt')

    torch.save(model, f'ckpts/cnn-trans/cnn-transformer-{dataset_name}-last.pt')
    plot_training(np.array(loss_list), np.array(acc_list), dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_epoch', type=int)
    parser.add_argument('--num_layers', type=int)
    args = parser.parse_args()

    random_seed = 3
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    DATASET_PATHS = [f'datasets/{args.dataset_name}']
    LR = args.lr
    EPOCHS = args.epoch
    WEIGHT_DECAY = args.weight_decay
    BATCH_SIZE = args.batch_size
    SAVE_EPOCH = args.save_epoch

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

    trainModel(DATASET_PATHS, LR, EPOCHS, WEIGHT_DECAY, BATCH_SIZE, SAVE_EPOCH, model_args)
