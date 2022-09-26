# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from ast import arg
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from torch.utils.data import DataLoader, RandomSampler,Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import re
import pprint
import os
import sys
sys.path.append(r'/home/dinglia/NER/BOND-master_release')
os.chdir(sys.path[-1])
print(os.getcwd())

from models.transformers import BertConfig,BertForTokenClassification,BertTokenizer,WEIGHTS_NAME
from models.transformers import RobertaConfig,RobertaForTokenClassification,RobertaTokenizer
#%%
# from models.transformers import WEIGHTS_NAME,BertForTokenClassification,BertTokenizer
# from models.transformers import RobertaForTokenClassification,RobertaTokenizer
import shutil
pattern = re.compile(r'\d+\n')
os.chdir(sys.path[0])
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

from model_utils import multi_source_label_refine, soft_frequency, mt_update, get_mt_loss, opt_grad,multi_source_label_refine_for_each_category
from eval import evaluate

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def initialize(args, model, t_total, epoch):

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
                eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if epoch == 0:
            if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")
            ):
                # Load in optimizer and scheduler states
                optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

        model.zero_grad()
        return model, optimizer, scheduler



def count_argsidence_for_each_category(logits,ground_truth_label,label_argsidence_map,label_number_map):

    softmax = torch.nn.Softmax(dim=2)
    softmax_logits=softmax(logits)# softmax_logits:[batch size,max seq len,num_labels]
    predict_label=torch.argmax(softmax_logits,dim=2) # predict_label:[batch size,max seq len]

    for i in range(0,predict_label.shape[0]):
        for j in range(1,predict_label.shape[1]):
            if ground_truth_label[i,j].tolist()!=pad_token_label_id:
                if predict_label[i,j]==ground_truth_label[i,j]: 
                    label_argsidence_map[predict_label[i,j].tolist()]+=softmax_logits[i,j,predict_label[i,j].tolist()].tolist() # 相应的类别加上预测得分
                    label_number_map[predict_label[i,j].tolist()]+=1
    return label_argsidence_map,label_number_map
    


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    label_argsidence_map={kk:0.0 for kk in range(len(label_map))}
    label_number_map={kk:0.0000001 for kk in range(len(label_map))}
    new_label_argsidence_map={kk:0.0 for kk in range(len(label_map))}
    if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(os.path.join(args.output_dir,'tfboard'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model, optimizer, scheduler = initialize(args, model, t_total, 0)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path): 
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]
    if args.mt: # args.mt=0
        teacher_model = model
    self_training_teacher_model = model

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # Update labels periodically after certain begin step
            if global_step >= args.self_training_begin_step:

                # Update a new teacher periodically
                delta = global_step - args.self_training_begin_step
                if delta % args.self_training_period == 0:
                    self_training_teacher_model = copy.deepcopy(model)
                    self_training_teacher_model.eval()
                    # total_num =sum(p.numel() for p in self_training_teacher_model.parameters())
                    if args.update_every_period:
                        """置信度计算"""
                        for label_id in label_argsidence_map.keys():
                            new_label_argsidence_map[label_id]=label_argsidence_map[label_id]/label_number_map[label_id]
                            print('{}的置信度阈值为：{}'.format(label_map[label_id],new_label_argsidence_map[label_id]))
                        label_argsidence_map={kk:0.0 for kk in range(len(label_map))}
                        label_number_map={kk:0.0000001 for kk in range(len(label_map))}
                        with open(os.path.join(args.data_dir,'argsidence.txt'),'a',encoding='utf-8') as f:
                            f.write(str(new_label_argsidence_map)+'\n')
                    else:
                        if delta==0:
                            """置信度计算"""
                            for label_id in label_argsidence_map.keys():
                                new_label_argsidence_map[label_id]=label_argsidence_map[label_id]/label_number_map[label_id]

                                print('{}的置信度阈值为：{}'.format(label_map[label_id],new_label_argsidence_map[label_id]))
                            with open(os.path.join(args.data_dir,'argsidence.txt'),'a',encoding='utf-8') as f:
                                f.write(str(new_label_argsidence_map)+'\n')
                    
                    # Re-initialize the student model once a new teacher is obtained  
                    if args.self_training_reinit:
                        model, optimizer, scheduler = initialize(args, model, t_total, 0)

                # Using current teacher to update the label 
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)

                softmax = torch.nn.Softmax(dim=2)
                with torch.no_grad():
                    outputs = self_training_teacher_model(**inputs)# # (loss), logits,sequence_output,(hidden_states), (attentions) 

                label_mask = None 
                if args.self_training_label_mode == "hard":
                    if args.whether_category_oriented:
                        pred_labels=softmax(outputs[0])
                        pred_labels, label_mask=multi_source_label_refine_for_each_category(args,new_label_argsidence_map,batch[3],pred_labels,pad_token_label_id)# label_mask:[batch size,max seq len],pred_labels[batch size,max seq len, num labels]
                        new_pred_labels=torch.argmax(pred_labels,dim=2)# new_pred_labels:[batch size,max seq len]
                    else:
                        pred_labels = torch.argmax(outputs[0], axis=2) # outputs[0]:[batch size,max seq len,num labels],pred_labels:[batch size,max seq len]
                        pred_labels, label_mask = multi_source_label_refine(args,batch[5],batch[3],pred_labels,pad_token_label_id,pred_logits=outputs[0])
                        new_pred_labels=pred_labels# new_pred_labels:[batch size,max seq len]
                elif args.self_training_label_mode == "soft": 
                    pred_labels = soft_frequency(logits=outputs[0], power=2) # pred_labels:[batch size,max seq len,num labels]
                    if args.whether_category_oriented:
                        raise NotImplementedError
                        # pred_labels, label_mask=multi_source_label_refine_for_each_category(args,new_label_argsidence_map,batch[3],pred_labels,pad_token_label_id)
                        # new_pred_labels=torch.argmax(pred_labels,dim=2)
                    else:
                        pred_labels, label_mask = multi_source_label_refine(args,batch[5],batch[3],pred_labels,pad_token_label_id) 
                        new_pred_labels=pred_labels # new_pred_labels：[batch size,max seq len,num labels]
                # batch[5]:all_hp_labels, batch[3]:all_labels。 

                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": new_pred_labels, "label_mask": label_mask}

            
            else:       
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )


            outputs = model(**inputs) # # (loss), logits, (hidden_states), (attentions) 
            loss, logits, final_embeds = outputs[0], outputs[1], outputs[2] # model outputs are always tuple in pytorch-transformers (see doc)
            # logits:[batch size,max seq len,num labels]
            # print(loss)

            """confidence threshold calculation"""
            # if global_step>=args.argsidence_threshold_begin_step:
            if global_step>=args.warmup_steps and global_step<args.self_training_begin_step:
                label_argsidence_map,label_number_map=count_argsidence_for_each_category(logits,batch[3],label_argsidence_map,label_number_map)

            mt_loss, vat_loss = 0, 0 # logits:[batch size, max seq len, num labels];final_emebeds:[batch size, max seq len,768]
            # region
            # Mean teacher training scheme
            if args.mt and global_step % args.mt_updatefreq == 0:
                    update_step = global_step // args.mt_updatefreq
                    if update_step == 1:
                        teacher_model = copy.deepcopy(model)
                        teacher_model.train(True)
                    elif update_step < args.mt_rampup:
                        alpha = args.mt_alpha1
                    else:
                        alpha = args.mt_alpha2
                    mt_update(teacher_model.named_parameters(), model.named_parameters(), args.mt_avg, alpha, update_step)

            if args.mt and update_step > 0:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**inputs)
                        teacher_logits, teacher_final_embeds = teacher_outputs[1], teacher_outputs[2]

                    _lambda = args.mt_lambda
                    if args.mt_class != 'smart':
                        _lambda = args.mt_lambda * min(1,math.exp(-5*(1-update_step/args.mt_rampup)**2))

                    if args.mt_loss_type == "embeds":
                        mt_loss = get_mt_loss(final_embeds, teacher_final_embeds.detach(), args.mt_class, _lambda)
                    else:
                        mt_loss = get_mt_loss(logits, teacher_logits.detach(), args.mt_class, _lambda)

            # Virtual adversarial training
            if args.vat:

                    if args.model_type in ["roberta", "camembert", "xlmroberta"]:
                        word_embed = model.roberta.get_input_embeddings()
                    elif args.model_type == "bert":
                        word_embed = model.bert.get_input_embeddings()
                    elif args.model_type == "distilbert":
                        word_embed = model.distilbert.get_input_embeddings()

                    if not word_embed:
                        print("Model type not supported. Unable to retrieve word embeddings.")
                    else:
                        embeds = word_embed(batch[0])
                        vat_embeds = (embeds.data.detach() + embeds.data.new(embeds.size()).normal_(0, 1)*1e-5).detach()
                        vat_embeds.requires_grad_()

                        vat_inputs = {"inputs_embeds": vat_embeds, "attention_mask": batch[1], "labels": batch[3]}
                        if args.model_type != "distilbert":
                            inputs["token_type_ids"] = (
                                batch[2] if args.model_type in ["bert", "xlnet"] else None
                            )  # XLM and RoBERTa don"t use segment_ids

                        vat_outputs = model(**vat_inputs)
                        vat_logits, vat_final_embeds = vat_outputs[1], vat_outputs[2]

                        if args.vat_loss_type == "embeds":
                            vat_loss = get_mt_loss(vat_final_embeds, final_embeds.detach(), args.mt_class, 1)
                        else:
                            vat_loss = get_mt_loss(vat_logits, logits.detach(), args.mt_class, 1)

                        vat_embeds.grad = opt_grad(vat_loss, vat_embeds, optimizer)[0]
                        norm = vat_embeds.grad.norm()

                        if (torch.isnan(norm) or torch.isinf(norm)):
                            print("Hit nan gradient in embed vat")
                        else:
                            adv_direct = vat_embeds.grad / (vat_embeds.grad.abs().max(-1, keepdim=True)[0]+1e-4)
                            vat_embeds = vat_embeds + args.vat_eps * adv_direct
                            vat_embeds = vat_embeds.detach()

                            vat_inputs = {"inputs_embeds": vat_embeds, "attention_mask": batch[1], "labels": batch[3]}
                            if args.model_type != "distilbert":
                                inputs["token_type_ids"] = (
                                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                                )  # XLM and RoBERTa don"t use segment_ids

                            vat_outputs = model(**vat_inputs)
                            vat_logits, vat_final_embeds = vat_outputs[1], vat_outputs[2]
                            if args.vat_loss_type == "embeds":
                                vat_loss = get_mt_loss(vat_final_embeds, final_embeds.detach(), args.mt_class, args.vat_lambda) \
                                        + get_mt_loss(final_embeds, vat_final_embeds.detach(), args.mt_class, args.vat_lambda)
                            else:
                                vat_loss = get_mt_loss(vat_logits, logits.detach(), args.mt_class, args.vat_lambda) \
                                        + get_mt_loss(logits, vat_logits.detach(), args.mt_class, args.vat_lambda)

            # endregion

            loss = loss + args.mt_beta * mt_loss + args.vat_beta * vat_loss
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:

                        logger.info("***** Entropy loss: %.4f, mean teacher loss : %.4f; vat loss: %.4f *****", \
                            loss - args.mt_beta * mt_loss - args.vat_beta * vat_loss, \
                            args.mt_beta * mt_loss, args.vat_beta * vat_loss)
                        results, _, best_dev, is_updated = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_dev, mode="dev", prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        results, _, best_test, is_updated  = evaluate(args, model, tokenizer, labels,pad_token_label_id, best_test, mode="dev", prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("test_{}".format(key), value, global_step)
                        output_dirs = []
                        if args.local_rank in [-1, 0] and is_updated:
                            updated_self_training_teacher = True
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-best"))

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))

                        if len(output_dirs) > 0:
                            for output_dir in output_dirs:
                                logger.info("Saving model checkpoint to %s", args.output_dir)
                                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                # They can then be reloaded using `from_pretrained()`
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                                
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model, global_step, tr_loss / global_step, best_dev, best_test


def parse_argument(parser):
    # Required parameters

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: "
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list:,"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )


    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--labels", default="", type=str,help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--label_index", type=int, default=1,help="the column index of label")
    parser.add_argument('--visible_device', default="0")
    parser.add_argument("--train_file", type=str, default="train.tsv", help="the file to train")
    parser.add_argument("--dev_file", type=str, default="dev.tsv", help="the file to evaluate")
    parser.add_argument("--test_file", type=str, default="test.tsv", help="the file to predict")


    # mean teacher
    parser.add_argument('--mt', type = int, default = 0, help = 'mean teacher.')
    parser.add_argument('--mt_updatefreq', type=int, default=1, help = 'mean teacher update frequency')
    parser.add_argument('--mt_class', type=str, default="kl", help = 'mean teacher class, choices:[smart, prob, logit, kl(default), distill].')
    parser.add_argument('--mt_lambda', type=float, default=1, help= "trade off parameter of the consistent loss.")
    parser.add_argument('--mt_rampup', type=int, default=300, help="rampup iteration.")
    parser.add_argument('--mt_alpha1', default=0.99, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_alpha2', default=0.995, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_beta', default=10, type=float, help="coefficient of mt_loss term.")
    parser.add_argument('--mt_avg', default="exponential", type=str, help="moving average method, choices:[exponentail(default), simple, double_ema].")
    parser.add_argument('--mt_loss_type', default="logits", type=str, help="subject to measure model difference, choices:[embeds, logits(default)].")

    # virtual adversarial training
    parser.add_argument('--vat', type = int, default = 0, help = 'virtual adversarial training.')
    parser.add_argument('--vat_eps', type = float, default = 1e-3, help = 'perturbation size for virtual adversarial training.')
    parser.add_argument('--vat_lambda', type = float, default = 1, help = 'trade off parameter for virtual adversarial training.')
    parser.add_argument('--vat_beta', type = float, default = 1, help = 'coefficient of the virtual adversarial training loss term.')
    parser.add_argument('--vat_loss_type', default="logits", type=str, help="subject to measure model difference, choices = [embeds, logits(default)].")

    # self-training
    parser.add_argument('--whether_self_training',action="store_true", help = 'whether to use teacher-student self training')
    parser.add_argument('--self_training_reinit', type = int, default = 0, help = 're-initialize the student model if the teacher model is updated.')
    parser.add_argument('--self_training_begin_step', type = int, default = 900, help = 'the begin step (usually after the first epoch) to start self-training.')
    parser.add_argument('--self_training_label_mode', type = str, default = "hard", help = 'pseudo label type. choices:[hard(default), soft].')
    parser.add_argument('--self_training_period', type = int, default = 878, help = 'the self-training period.')
    parser.add_argument('--self_training_hp_label', type = float, default = -1, help = 'use high precision label.')
    parser.add_argument('--self_training_hp_label_category', type = int, default = -1, help = 'for category oriented self training ,decide whether to add false labels to loss calculation,choices:[0(not included),1(included)]')
    parser.add_argument('--self_training_ensemble_label', type = int, default = 0, help = 'use ensemble label.')
    parser.add_argument("--whether_category_oriented", action="store_true",help="whether to use different category distribution for psdudo label")
    parser.add_argument("--argsidence_test", action="store_true",help="whether to use argsidence test code")
    parser.add_argument("--update_every_period", action="store_true",help="whether update argsidence every self training period")


    # Use data from weak.json
    parser.add_argument('--load_weak', action="store_true", help = 'Load data from weak.json.')
    parser.add_argument('--remove_labels_from_weak', action="store_true", help = 'Use data from weak.json, and remove their labels for semi-supervised learning')
    parser.add_argument('--rep_train_against_weak', type = int, default = 1, help = 'Upsampling training data again weak data. Default: 1')

    args = parser.parse_args()
#     args=parser.parse_args("--data_dir /data/dlp/Pesudo_Label_Learning/BOND-master/scripts/../dataset/conll03_distant/ \
# --model_type bert \
# --model_name_or_path roberta-base \
# --learning_rate 0.1 \
# --weight_decay 1e-4 \
# --adam_epsilon 1e-8 \
# --adam_beta1 0.9 \
# --adam_beta2 0.98 \
# --num_train_epochs 5 \
# --warmup_steps 200 \
# --per_gpu_train_batch_size 16 \
# --per_gpu_eval_batch_size 32 \
# --logging_steps 100 \
# --save_steps 100000 \
# --do_train \
# --do_eval \
# --do_predict \
# --evaluate_during_training \
# --output_dir output \
# --seed 42 \
# --max_seq_length 128 \
# --overwrite_output_dir \
# --self_training_reinit 0 \
# --self_training_begin_step 500 \
# --self_training_label_mode 1 \
# --self_training_period 200 \
# --self_training_hp_label_category 1 \
# --visible_device 0 \
# --no_cuda \
# --label_index 1".split())

    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args=parse_argument(parser)

    print("The path of the dataest: {}".format(args.data_dir))
    confidence_file=os.path.join(args.data_dir,'confidence.txt')
    if os.path.exists(confidence_file):
        os.remove(confidence_file)

    MODEL_CLASSES = {
        "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
        "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    }
    # 删除output文件夹
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        print('delete output dir')
        shutil.rmtree(args.output_dir)


    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


    from data_utils_conll import load_and_cache_examples,get_labels
    labels = get_labels(args.data_dir)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    label_map={kk:labels[kk] for kk in range(len(labels))}
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training

    set_seed(args)
    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
    # import ipdb; ipdb.set_trace()
    if args.load_weak:
        weak_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="weak", remove_labels=args.remove_labels_from_weak)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset]*args.rep_train_against_weak + [weak_dataset,])

    model = model_class.from_pretrained(args.model_name_or_path,
                                    from_tf=bool(".ckpt" in args.model_name_or_path),
                                    config=config,
                                    cache_dir=args.cache_dir if args.cache_dir else None)

    model.to(args.device)
    # check number of parameters
    total_num =sum(p.numel() for p in model.parameters())
    print('number of model parameters:',total_num)
    # check the model architecture
    print('*****'*10)
    # print(model)
    print('*****'*10)
    model, global_step, tr_loss, best_dev, best_test=train(args, train_dataset, model, tokenizer, labels,pad_token_label_id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving last-practice: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt")) # 这好像并不是效果最好的模型，而是最后一个模型

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        if not best_dev:
            best_dev = [0, 0, 0]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _, best_dev, _ = evaluate(args, model, tokenizer, labels,pad_token_label_id, best=best_dev, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir,'checkpoint-best'), do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(os.path.join(args.output_dir,'checkpoint-best'))
        model.to(args.device)

        if not best_test:
            best_test = [0, 0, 0]
        result, predictions, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_test, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir,'checkpoint-best', "{}_test_results.txt".format(args.test_file.split('.')[0]))
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.json"), "r") as f:
                example_id = 0
                data = json.load(f)
                for item in data:
                    output_line = str(item["str_words"]) + " " + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                    example_id += 1