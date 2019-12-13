We used the baseline code for SQLNEt as implemented by Xu et al. SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning. This repository has the code changes made for the using Bert Embeddings.

Files Modified:

/sqlnet/utils.py

/extract_vocab.py

/sqlnet/model/modules/word_embedding.py

Versions:
Python 3.6.9
Pytorch 1.3.1 




# SQLNet

This repo provides an implementation of SQLNet and Seq2SQL neural networks for predicting SQL queries on [WikiSQL dataset](https://github.com/salesforce/WikiSQL). The paper is available at [here](https://arxiv.org/abs/1711.04436).

## Citation

> Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning.

## Bibtex

```
@article{xu2017sqlnet,
  title={SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning},
  author={Xu, Xiaojun and Liu, Chang and Song, Dawn},
  journal={arXiv preprint arXiv:1711.04436},
  year={2017}
}
```

## Installation
The data is in `data.tar.bz2`. Unzip the code by running
```bash
tar -xjvf data.tar.bz2
```

The code is written using PyTorch in Python 3.6.9. Check [here](http://pytorch.org/) to install PyTorch. You can install other dependency by running 
```bash
pip install -r requirements.txt
```
Additional Requirements for Bert:

pip install bert-embedding
If you want to run on GPU machine, please install `mxnet-cu92`.
pip install mxnet-cu92

## Extract the bert embedding for training.
Run the following command to process the pretrained glove embedding for training the word embedding:
```bash
python extract_vocab.py
```

## Train
The training script is `train.py`. To see the detailed parameters for running:
```bash
python train.py -h
```

Some typical usage are listed as below:

Train a SQLNet model with column attention:
```bash
python train.py --ca
```

Train a SQLNet model with column attention and trainable embedding (requires pretraining without training embedding, i.e., executing the command above):
```bash
python train.py --ca --train_emb
```


## Test
The script for evaluation on the dev split and test split. The parameters for evaluation is roughly the same as the one used for training. For example, the commands for evaluating the models from above commands are:

Test a trained SQLNet model with column attention
```bash
python test.py --ca
```

Test a trained SQLNet model with column attention and trainable embedding:
```bash
python test.py --ca --train_emb
```



