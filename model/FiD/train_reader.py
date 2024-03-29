import time
import os 
import sys
import wandb
import torch
import transformers
import numpy as np
import subprocess
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
from knockknock import slack_sender
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torch.multiprocessing as mp


import src.slurm
import src.util
import src.evaluation
import src.data
import src.model


@slack_sender(webhook_url="https://hooks.slack.com/services/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", channel="knock_knock")
def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, rank, collator, best_dev_em, checkpoint_path, logger, tokenizer, run=None):
    torch.manual_seed(0) #different seed for different sampling depending on global_rank
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas = opt.world_size,
        rank = rank
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=4,
        collate_fn=collator,
        shuffle = False,
        pin_memory = True
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.eval()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if opt.do_wandb_log:
                        run.log({"steps": step, "train_loss": curr_loss/opt.eval_freq, "dev_EM": 100*dev_em, "lr": scheduler.get_last_lr()[0]})
                    curr_loss = 0
                model.train()

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

def main_loop(gpu, opt):
    run = None
    rank = opt.nr * opt.gpus + gpu
    opt.is_main = (rank == 0)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size = opt.world_size,
        rank=rank
    )

    if opt.is_main and opt.do_wandb_log:
        run = wandb.init(
            name = opt.name,
            entity = opt.entity,
            project = opt.project,
        )
    torch.manual_seed(0)
    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)
    opt.device = gpu
    torch.cuda.set_device(gpu)
    if opt.fine_tune_pretrained_model:
        step, best_dev_em = 0, 0.0
        model, optimizer, scheduler = src.util.load_with_pretrained_model(model_class, opt.model_path, opt)
        if opt.is_distributed:
            model = model.to(gpu)
        logger.info(f"Pretrained Model loaded from {opt.model_path}")
    else:
        if not checkpoint_exists and opt.model_path == "none":
            t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
            model = src.model.FiDT5(t5.config)
            model.load_t5(t5.state_dict())
            optimizer, scheduler = src.util.set_optim(opt, model)
            step, best_dev_em = 0, 0.0
        elif opt.model_path == "none":
            load_path = checkpoint_path / 'checkpoint' / 'latest'
            model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
                src.util.load(model_class, load_path, opt, reset_params=False)
            logger.info(f"Model loaded from {load_path}")
        else:
            model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
                src.util.load(model_class, opt.model_path, opt, reset_params=True)
            logger.info(f"Model loaded from {opt.model_path}")
        if opt.is_distributed:
            model = model.to(gpu)
        logger.info(f"Training from scratch")

    
    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu]
        )

    logger.info("Start training")

    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        rank,
        collator,
        best_dev_em,
        checkpoint_path,
        logger,
        tokenizer,
        run
    )


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    options.add_DDP_options()
    options.add_wandb_options()
    opt = options.parse()

    if opt.gpus > 1:
        opt.is_distributed = True
    opt.world_size = opt.gpus * opt.nodes
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    main_addr = hostnames.split()[0].decode('utf-8')
    os.environ["MASTER_ADDR"] = main_addr
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(main_loop, nprocs=opt.gpus, args=(opt, ))

