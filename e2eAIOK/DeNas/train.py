import sys
import os
from pathlib import Path
import argparse
import numpy as np
import torch
import random
import yaml
from easydict import EasyDict as edict
import sentencepiece as sp
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from e2eAIOK.common.trainer.torch_trainer import TorchTrainer 
import e2eAIOK.common.trainer.utils.utils as utils
from e2eAIOK.common.trainer.model.model_builder_asr import ModelBuilderASR
from e2eAIOK.common.trainer.model.model_builder_cv import ModelBuilderCV
from e2eAIOK.common.trainer.model.model_builder_nlp import ModelBuilderNLP
from e2eAIOK.common.trainer.data.data_builder_librispeech import DataBuilderLibriSpeech
from e2eAIOK.common.trainer.data.data_builder_cifar import DataBuilderCIFAR
from e2eAIOK.common.trainer.data.data_builder_nlp import DataBuilderNLP
from e2eAIOK.common.trainer.data.data_builder_cv import DataBuilderCV
from e2eAIOK.common.trainer.data.data_builder_squad import DataBuilderSQuAD
from asr.asr_trainer import ASRTrainer
from asr.trainer.schedulers import NoamScheduler
from asr.trainer.losses import ctc_loss, kldiv_loss
from asr.utils.metric_stats import ErrorRateStats
from cv.cv_trainer import CVTrainer
from nlp.utils import bert_create_optimizer, bert_create_criterion, bert_create_scheduler, bert_create_metric
from nlp.bert_trainer import BERTTrainer


def parse_args(args):
    parser = argparse.ArgumentParser('Torch model training or evluation............')
    parser.add_argument('--domain', type=str, default=None, choices=['cnn','vit','bert','asr'], help='training model domain')
    parser.add_argument('--conf', type=str, default=None, help='training or evluation conf file')
    parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for consistent training')
    train_args = parser.parse_args(args)
    return train_args

def main(args):
    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
    root_dir = Path(os.getcwd()).parent.parent
    conf_file = os.path.join(root_dir, os.path.basename(args.conf))
    with open(conf_file) as f:
        cfg = edict(yaml.safe_load(f))

    ext_dist.init_distributed(backend=cfg.dist_backend)

    if args.domain in ['cnn','vit']:
        model = ModelBuilderCV(cfg).create_model()
        train_dataloader, eval_dataloader = DataBuilderCIFAR(cfg).get_dataloader()
        optimizer = utils.create_optimizer(model, cfg)
        criterion = utils.create_criterion(cfg)
        scheduler = utils.create_scheduler(optimizer, cfg)
        metric = utils.create_metric(cfg)
        trainer = CVTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric)
    elif args.domain == 'bert':
        model = ModelBuilderNLP(cfg).create_model()
        train_dataloader, eval_dataloader, other_data = DataBuilderSQuAD(cfg).get_dataloader()
        optimizer = bert_create_optimizer(model, cfg)
        criterion = bert_create_criterion(cfg)
        scheduler = bert_create_scheduler(cfg)
        metric = bert_create_metric(cfg)
        trainer = BERTTrainer(cfg, model, train_dataloader, eval_dataloader, other_data, optimizer, criterion, scheduler, metric)
    elif args.domain == 'asr':
        if cfg.structure:
            model = ModelBuilderASR(cfg).create_model()
        else:
            model = ModelBuilderASR(cfg).load_pretrained_model_and_prune()
        tokenizer = sp.SentencePieceProcessor()
        train_dataloader, eval_dataloader = DataBuilderLibriSpeech(cfg, tokenizer).get_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr_adam"], betas=(0.9, 0.98), eps=0.000000001)
        criterion = {"ctc_loss": ctc_loss, "seq_loss": kldiv_loss}
        scheduler = NoamScheduler(lr_initial=cfg["lr_adam"], n_warmup_steps=cfg["n_warmup_steps"])
        metric = ErrorRateStats()
        trainer = ASRTrainer(cfg, model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, metric, tokenizer)
    else:
        raise RuntimeError(f"Domain {args.domain} is not supported")
    trainer.fit()

if __name__ == '__main__':
    args  = parse_args(sys.argv[1:])
    main(args)