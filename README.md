# End-to-End Speech Emotion Recognition based on CNN-Transformer

End-to-End **S**peech **E**motion **R**ecognition based on **C**NN-**T**ransformer (**SERCT**) is a model desinged to recognize emotions from speech signals using original audio input.

<h1 align="center">
    <img src="assets/banner.webp", alt="" />
</h1>

## Contents

- [Install](#install)
- [Reproduction](#reproduction)
- [Experiments](#experiments)
- [Application](#application)

## Install

1. Clone this repository and navigate to the SERCT folder
```bash
git clone https://github.com/0105zhengsiqi/SERCT.git
cd SERCT
```

2. Install Package
```bash
pip install -r requirements.txt
```

## Reproduction

For simplicity, we place the training code and testing code in the same script named [cnn_trans.sh](./scripts/cnn_trans.sh). Before starting training, it is necessary to prepare **training data** and **GPU resources**.

### Data Preparation

For training data, use the following format:
```python
datasets
|-- EMO-DB_de
|   `-- test
|       |-- angry
|       |-- disgusted
|       |-- fearful
|       |-- happy
|       |-- neutral
|       |-- sad
|       `-- surprised
|-- ESD
|   |-- test
|   |   |-- angry
|   |   |-- happy
|   |   |-- neutral
|   |   |-- sad
|   |   `-- surprised
|   `-- train
|       |-- angry
|       |-- happy
|       |-- neutral
|       |-- sad
|       `-- surprised
|-- ESD_zh
|   |-- test
|   |   |-- angry
|   |   |-- disgusted
|   |   |-- fearful
|   |   |-- happy
|   |   |-- neutral
|   |   |-- sad
|   |   `-- surprised
|   `-- train
|       |-- angry
|       |-- disgusted
|       |-- fearful
|       |-- happy
|       |-- neutral
|       |-- sad
|       `-- surprised
`-- TESS
    |-- test
    |   |-- angry
    |   |-- disgusted
    |   |-- fearful
    |   |-- happy
    |   |-- neutral
    |   |-- sad
    |   `-- surprised
    `-- train
        |-- angry
        |-- disgusted
        |-- fearful
        |-- happy
        |-- neutral
        |-- sad
        `-- surprised
```

### Train and Eval

Just run the following command:
```bash
bash scripts/cnn_trans.sh
```

## Experiments

For baselines, please run the following code one by one.
```python
# for cnn
python exps/cnn/train.py

# for cnn-bilstm
python exps/cnn_bilstm/train.py

# for lstm
python exps/lstm/train.py

# for mlp
python exps/mlp/train.py

# for svm
python exps/svm
```

## Application

To run the demonstrator, please follow the following steps.
1. Put the ckpt in the `app/ckpt` directory.
2. `cd app`
3. `streamlit run main.py`