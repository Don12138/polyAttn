import argparse
import logging
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import xavier_uniform_
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from utils import parsing
from utils.data_utils import PolymerDataset
from utils.train_utils import get_lr, grad_norm, NoamLR, param_count, param_norm, set_seed, setup_logger, load_data, batch_force_reshape, MTAdam
from models.polygnn import polygnn
import pdb
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter




def get_train_parser():
    parser = argparse.ArgumentParser("train")
    parsing.add_common_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def main(args):
    parsing.log_args(args)


    # ----------------------------------------------------------
    # ---------------------initialization-----------------------
    # ----------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.mpn_type == "polygnn":
        model_class = polygnn
    else:
        raise ValueError(f"Model {args.model} not supported!")

    # ----------------------------------------------------------
    # --------------hyperparameter-optimization-----------------
    # ----------------------------------------------------------

    # 需要进行超参调优
    hps = {
            "ffn_capacity" : 2,
            "depth":3,
            "readout_dim": 128,
            "activation": nn.functional.leaky_relu
    }
    model = model_class(133,147,hps)



    if args.load_from:
        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

    model.to(device)
    model.train()


    logging.info(model)
    logging.info(f"Number of parameters = {param_count(model)}")



    # ----------------------------------------------------------
    # -----------optimizer/scheduler/loss-function--------------
    # ----------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    # optimizer = MTAdam(
    #     model.parameters(),lr=args.lr
    # )


    scheduler = NoamLR(
        optimizer,
        model_size=hps['readout_dim'],      # 有待修改
        warmup_steps=args.warmup_steps
    )

    loss_fn = nn.functional.mse_loss

    # ----------------------------------------------------------
    # ---------------------data---------------------------------
    # ----------------------------------------------------------

    train_dataset = PolymerDataset(f"/home/chenlidong/polyAttn/preprocessed/{args.data_name}",phase="train")
    valid_dataset = PolymerDataset(f"/home/chenlidong/polyAttn/preprocessed/{args.data_name}",phase="val")
    logging.info(f"Loaded {train_dataset.len()} train data and {valid_dataset.len()} validation data")

    total_step = 0
    accum = 0
    
    # ----------------------------------------------------------
    # ---------------------train--------------------------------
    # ----------------------------------------------------------

    o_start = time.time()
    logging.info("Start training")
    for epoch in range(args.epoch):
        logging.info(f"Start epoch {epoch}")
        model.zero_grad()
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory=True
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.valid_batch_size,
            shuffle=True,
            pin_memory=True
        )
        logging.info(f"Loaded {len(train_loader)} train batch and {len(valid_loader)} validation batch")

        # pdb.set_trace()
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            # if total_step >= 100:
            
            if total_step > args.max_steps:
                logging.info("Max steps reached, finish training")
                exit(0)

            batch = batch.to(device)

            tgt_ea = batch['ea']
            tgt_ip = batch['ip']
            pdb.set_trace()
            pred_ea = model(batch)

            loss = torch.sqrt(loss_fn(pred_ea,tgt_ea))
            # loss_1 = loss_fn_sum(force.flatten(),tgt_force)
            # loss = 0.8 * loss_0 + 0.2 * loss_1
            # losses = [loss_0, loss_1]


            # tensorboard
            # writer.add_scalar("train_loss", loss.item(), global_step=total_step, walltime= time.time()-o_start)
            # writer.add_scalar("train_energy_loss", loss_0.item(), global_step=total_step, walltime= time.time()-o_start)
            # writer.add_scalar("train_force_loss", loss_1.item(), global_step=total_step, walltime= time.time()-o_start)

            # with torch.autograd.detect_anomaly():
            # # 切换autocast需要修改下面的代码
            # # scaler.scale(loss).backward()  
            #     loss.backward()

            losses.append(loss.item())
            # losses_0.append(loss_0.item())
            # losses_1.append(loss_1.item())
            loss.backward()
            accum += 1


            if accum == args.accumulation_count:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

                # optimizer.step(losses,[1,1],None) # MToptimizer
                optimizer.step()
                scheduler.step()
                g_norm = grad_norm(model)
                model.zero_grad()
                accum = 0
                total_step += 1
                optimizer.zero_grad()
                writer.add_scalar("ea_loss", loss, global_step=total_step, walltime= time.time()-o_start)


            if (accum == 0) and (total_step > 0) and (total_step % args.log_iter == 0):
                logging.info(f"Step {total_step}, loss: {np.mean(losses)}, "
                            #  f"energy loss {np.mean(losses_0)}, force loss {np.mean(losses_1)}, "
                             f"p_norm: {param_norm(model)}, g_norm: {g_norm}, "
                             f"lr: {get_lr(optimizer): .6f}, elapsed time: {time.time() - o_start: .0f}")
                sys.stdout.flush()
                losses = []
                # losses_0 = []
                # losses_1 = []
    # ----------------------------------------------------------
    # ---------------------eval---------------------------------
    # ----------------------------------------------------------
            if (accum == 0) and (total_step > 0) and (total_step % args.eval_iter == 0):
                model.eval()
                eval_count = valid_dataset.len() // args.valid_batch_size - 1
                loss = 0.0
                # loss_0 = 0.0
                # loss_1 = 0.0
                for eval_idx, eval_batch in enumerate(valid_loader):
                    if eval_idx >= eval_count:
                        break
                    eval_batch = eval_batch.to(device)
                    tgt_ea = eval_batch['ea']
                    tgt_ip = eval_batch['ip']
                    
                    pred_ea = model(eval_batch)

                    eval_loss = torch.sqrt(loss_fn(pred_ea,tgt_ea))
                    

                    # eval_loss = eval_loss_0
                    loss += eval_loss.item() / eval_count

                writer.add_scalar("val_loss", loss, global_step=total_step, walltime= time.time()-o_start)
                # writer.add_scalar("val_energy_loss", loss_0, global_step=total_step, walltime= time.time()-o_start)
                # writer.add_scalar("val_force_loss", loss_1, global_step=total_step, walltime= time.time()-o_start)
                logging.info(f"Evaluation (with teacher) at step {total_step}, eval loss: {loss}")
                sys.stdout.flush()

                model.train()

            if (accum == 0) and (total_step > 0) and (total_step % args.save_iter == 0):
                n_iter = total_step // args.save_iter - 1

                logging.info(f"Saving at step {total_step}")
                sys.stdout.flush()
                state = {
                    "args": args,
                    "state_dict": model.state_dict()
                }
                torch.save(state, os.path.join(args.save_dir, f"model.{total_step}_{n_iter}.pt"))

        # lastly
        if (args.accumulation_count > 1) and (accum > 0):
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            scheduler.step()
            model.zero_grad()
            accum = 0


if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()
    
    # 自动检测梯度错误的现象
    torch.autograd.set_detect_anomaly(True)
    # 加速训练过程，节省内存
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True 

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)
    # writer = SummaryWriter(f'/home/chenlidong/polyAttn/tensorboard/{args.exp_no}')
    # profile的打印程度
    torch.set_printoptions(profile="default")
    main(args)
