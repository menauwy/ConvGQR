#from IPython import embed
import logging
import sys
sys.path.append('..')
sys.path.append('.')
import os
from os import path
print(os.environ.get('CONDA_DEFAULT_ENV'))
print(os.environ.get('PYTHONPATH'))
print(sys.executable)
import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
# import toml

#from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration
#from torch.utils.tensorboard import SummaryWriter

from models import load_model
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer, print_res
from data_structure import T5RewriterIRDataset_qrecc, T5RewriterIRDataset_topiocqa
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def save_model(args, model, query_tokenizer, save_model_order, epoch, step, loss):
    """Deprecated: use general version in below"""
    output_dir = os.path.join(args.model_output_path, '{}-{}-best-model'.format("KD-ANCE-prefix", args.decode_type))
    check_dir_exist_or_build([output_dir])

    model_to_save = model.module if hasattr(model, 'module') else model
    #model_to_save.t5.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

def save_checkpoint(args, model, query_tokenizer, save_model_order, epoch, step, loss, optimizer, scheduler):
    """
    To continue training a checkpoint, save optimizer, eposh, step and loss
    """
    # output_dir = os.path.join(args.model_output_path, '{}-{}-best-model'.format("KD-ANCE-prefix", args.decode_type))
    output_dir = os.path.join(args.model_output_path, '{}-{}-best-model-checkpoint'.format("KD-ANCE-prefix", args.decode_type))

    check_dir_exist_or_build([output_dir])
    checkpoint_file = output_dir + "/checkpoint.pt"

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    # save optimizer and scheduler states, epoch and step.
    torch.save({
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_file)
    #logger.info("Epoch {}, Step {}, Save checkpoint at {}".format(epoch, step, output_dir))

def load_checkpoint(args, optimizer, scheduler, logger):
    """
    To continue training a checkpoint, load optimizer, epoch, step and loss
    """
    output_dir = os.path.join(args.model_output_path, '{}-{}-best-model'.format("KD-ANCE-prefix", args.decode_type))
    checkpoint_file = output_dir + "/checkpoint.pt"
    # load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(output_dir)
    model = T5ForConditionalGeneration.from_pretrained(output_dir).to(args.device)

    # load optimizer and scheduler states, epoch and step.
    checkpoint = torch.load(checkpoint_file)
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #logger.info("Eposh {}, Step {}, Load checkpoint from {}".format(epoch, step, output_dir))
    return model, tokenizer, epoch, step, loss, optimizer, scheduler

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_kd_loss(query_embs, kd_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, kd_embs)


def train(args, logger):
    passage_tokenizer, passage_encoder = load_model("ANCE_Passage", args.pretrained_passage_encoder)
    passage_encoder = passage_encoder.to(args.device)
   
    query_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_query_encoder)
    query_encoder = T5ForConditionalGeneration.from_pretrained(args.pretrained_query_encoder).to(args.device)
    
    # gpu setting
    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    if args.train_dataset == "qrecc":
        train_dataset = T5RewriterIRDataset_qrecc(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset == "topiocqa":
        train_dataset = T5RewriterIRDataset_topiocqa(args, query_tokenizer, args.train_file_path)
    
    train_loader = DataLoader(train_dataset, 
                                #sampler=train_sampler,
                                batch_size = args.batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args))

    logger.info("train dataset samples num = {}".format(len(train_dataset)))
    
    num_steps_per_epoch = (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    total_training_steps = args.num_train_epochs * num_steps_per_epoch
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    #local_step = 0 #only useful for training from checkpoint
    global_step = 0
    save_model_order = 0


    # check if train from checkpoint:
    if args.train_from_checkpoint:
        logger.info("Loading checkpoint...")
        query_encoder, query_tokenizer, start_epoch, last_step, loss, optimizer, scheduler = load_checkpoint(args, optimizer, scheduler, logger)
        logger.info("Load checkpoint, epoch = {}, local_step = {}, loss = {}".format(start_epoch, last_step, loss))
        start_step = last_step + 1
    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    if not args.train_from_checkpoint: #train from scratch
        # iterable object with tqdm progress bar
        epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
        #best_loss = float('inf') #else loaded from checkpoint
        #batch_iterator = enumerate(tqdm(train_loader,  desc="Step", disable=args.disable_tqdm))
    else:
        # iterable object start from checkpoint epoch
        epoch_iterator = trange(start_epoch, args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
        # Problem is: the loss can be higher again in a new epoch.
        #best_loss = float('inf') #loss
        #batch_iterator = enumerate(tqdm(train_loader,  desc="Step", disable=args.disable_tqdm), start=start_step)

    for epoch in epoch_iterator:
        query_encoder.train()
        passage_encoder.eval()
        best_loss = float('inf')
        for step, batch in enumerate(tqdm(train_loader,  desc="Step", disable=args.disable_tqdm)):
            # if the current setp is less than the desired resume_step, skip it to the next batch
            # if args.train_from_checkpoint and step < start_step:
            #     #logger.info("Step = {}, Batch ids = {}".format(step, batch['bt_sample_ids']))
            #     continue

            query_encoder.zero_grad()

            bt_conv_query = batch['bt_input_ids'].to(args.device) # B * len
            bt_conv_query_mask = batch['bt_attention_mask'].to(args.device)
            bt_pos_docs = batch['bt_pos_docs'].to(args.device) # B * len one pos
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_docs = batch['bt_neg_docs'].to(args.device) # B * len batch size negs
            bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
            bt_oracle_query = batch['bt_labels'].to(args.device)

            #logger.info("Batch ids = {}".format(batch['bt_sample_ids']))
            
            # https://huggingface.co/docs/transformers/main/en/model_doc/t5
            output = query_encoder(input_ids=bt_conv_query, 
                         attention_mask=bt_conv_query_mask, 
                         labels=bt_oracle_query)
            decode_loss = output.loss  # B * dim
            # query embeddings, why just first dimension?
            conv_query_embs = output.encoder_last_hidden_state[:, 0]

            with torch.no_grad():
                # freeze passage encoder's parameters
                pos_doc_embs = passage_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                #neg_doc_embs = passage_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # B * dim, hard negative

            #ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
            # In the paper, they use MSE loss of query and relevant doc embeddings as the retreival signal, inserted in training loss.
            ranking_loss = cal_kd_loss(conv_query_embs, pos_doc_embs) #MSE
            loss = decode_loss + args.alpha * ranking_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, local Step = {}, ranking loss = {}, decode loss = {}, total loss = {}".format(
                                epoch, # original to be epoch + 1
                                step,
                                ranking_loss.item(),
                                decode_loss.item(),
                                loss.item()))

            #log_writer.add_scalar("train_ranking_loss, decode_loss, total_loss", ranking_loss, decode_loss, loss, global_step)
            
            # previously here to be global_step += 1
            #local_step = 0 # reset local step for next epoch
            # save model finally
            if best_loss > loss:
                #save_model(args, query_encoder, query_tokenizer, save_model_order, epoch, global_step, loss.item())
                #logger.info("Get better loss at local step {}".format(step))
                save_checkpoint(args, query_encoder, query_tokenizer, save_model_order, epoch, step, loss.item(), optimizer, scheduler)
                best_loss = loss
                logger.info("Saved checkpoint: Epoch = {}, Local Step = {}, ranking loss = {}, decode loss = {}, total loss = {}".format(
                                epoch,
                                step,
                                ranking_loss.item(),
                                decode_loss.item(),
                                loss.item()))
            if not args.train_from_checkpoint: #from scratch
                global_step += 1 # avoid saving the model of the first step.
            else:
                global_step = step + epoch * num_steps_per_epoch

    logger.info("Training finish!")          
         


def get_args():
    parser = argparse.ArgumentParser()
    dataset_dir = '/home/wangym/data1/dataset/'
    model_dir = '/home/wangym/data1/model/'

    parser.add_argument("--train_from_checkpoint", action='store_true') #if specified in command line, then true. Defalut false
    parser.add_argument("--pretrained_query_encoder", type=str, default= model_dir + "pretrained/t5-base") # default="checkpoints/T5-base"
    parser.add_argument("--pretrained_passage_encoder", type=str, default= model_dir + "pretrained/ance-msmarco-passage") # default="checkpoints/ad-hoc-ance-msmarco"

    parser.add_argument("--train_dataset", type=str, default="qrecc") #qrecc for rewriter&expansion, plus topiocqa for expansion; the same as decode_type
    parser.add_argument("--train_file_path", type=str, default=dataset_dir + "qrecc/new_preprocessed/train_with_doc.json")
    parser.add_argument("--log_dir_path", type=str, default=model_dir+ "convgqr/train_qrecc")
    parser.add_argument('--model_output_path', type=str, default=model_dir+"convgqr/train_qrecc")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train")
    parser.add_argument("--decode_type", type=str, default="oracle") # "oracle" for rewrite and "answer" for expansion
    parser.add_argument("--use_prefix", type=bool, default=True)

    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=8)
    parser.add_argument("--use_data_percent", type=float, default=1)
    
    parser.add_argument("--num_train_epochs", type=int, default=15, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)

    parser.add_argument("--print_steps", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#, args.local_rank)
    args.device = device

    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    #log_writer = SummaryWriter(log_dir = args.log_dir_path)
    logging.basicConfig(filename=args.log_dir_path,level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) #create a logger with the name of the current module
    train(args, logger)
    # log_writer.close()
