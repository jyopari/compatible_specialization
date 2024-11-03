import sys
import os
import argparse
import torch
import numpy as np
from pprint import pprint


def parse_args():
    ############################################################################################
    ######################################### Parse args #######################################
    ############################################################################################

    #datasets      = ['openwebtext', 'tinystories']
    optims         = ['agdv2', 'adamw', 'sgd', 'lion', 'adamwn']
    poolings       = ['avg', 'last']
    routings       = ['standard', 'rpo']

    parser = argparse.ArgumentParser()
    # system
    parser.add_argument('--logdir',       type=str                  )
    parser.add_argument('--workers',      type=int,   default=8     )
    parser.add_argument('--distributed',  action='store_true'       )
    parser.add_argument('--seed',         type=int,   default=0     )
    parser.add_argument("--world-size",   default=1, type=int       )
    parser.add_argument("--dist-url",     default="env://", type=str)
    # architecture
    parser.add_argument('--arch',         type=str,   default=None  )
    parser.add_argument('--depth',        type=int,   default=6     )
    parser.add_argument('--width',        type=int,   default=384   )
    parser.add_argument('--context',      type=int,   default=256   )
    parser.add_argument('--heads',        type=int,   default=6     )
    parser.add_argument('--mlp',          type=str,   default='default')
    parser.add_argument('--pretrained',   action='store_true'       )
    parser.add_argument('--chkpt',        type=str,   default=None  )
    # router
    parser.add_argument('--router_method',type=str,   default=None,      choices=routings)
    parser.add_argument('--lb_scale',     type=float, default=0.0   ) # load balancing scale
    parser.add_argument('--topk',        type=int,   default=1      ) # topk routing
    parser.add_argument('--single_router', action='store_true'      )
    parser.add_argument('--router_ckpts', type=str,   nargs='+'     )
    parser.add_argument('--multi_layer_router', action='store_true' )
    parser.add_argument('--num_multi_layer_experts',type=int,                   default=1)
    # data
    #parser.add_argument('--dataset',      type=str,   default='cifar10',  choices=datasets)
    parser.add_argument('--datasets',     type=str,   nargs='+'     )
    parser.add_argument('--vocab_size',   type=int,   default=50257 )
    parser.add_argument('--batch_size',   type=int,   default=128   )
    parser.add_argument('--train_sep',    action='store_true')
    # training
    parser.add_argument('--optim',        type=str,   default='adamw',    choices=optims)
    parser.add_argument('--train_tokens', type=int,   default=100000)
    parser.add_argument('--grad_accum',   type=int,   default=1     )
    parser.add_argument('--lr',           type=float, default=1e-3  )
    parser.add_argument('--beta',         type=float, default=0.9   )
    parser.add_argument('--beta2',        type=float, default=0.99  )
    parser.add_argument('--wd',           type=float, default=0.01  )
    parser.add_argument('--clip',         action='store_true'       )
    parser.add_argument('--train_mlp',    action='store_true'       )
    parser.add_argument('--out_dir',      type=str,   default='out' )
    # logging
    parser.add_argument('--wandb',        action='store_true'       )
    parser.add_argument('--name',         type=str,   default='')
    parser.add_argument('--log_freq',     type=int,   default=200   )
    parser.add_argument('--eval_iter',    type=int,   default=50    )
    parser.add_argument('--eval',         action='store_true'       )
    parser.add_argument('--eval_sep',     type=str,   nargs='+'     )
    # analysis
    parser.add_argument('--analyze',      action='store_true'       )
    # analysis batch 
    parser.add_argument('--chkpt_base_name', type=str,  nargs='+'   )
    parser.add_argument('--max_step',     type=int,   default=8500  )
    parser.add_argument('--step_size',   type=int,   default=500    )
    parser.add_argument('--file_name',    type=str, default=None    ) 
    args = parser.parse_args()

    ############################################################################################
    ##################################### Set random seed ######################################
    ############################################################################################

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ############################################################################################
    ####################################### Print args #########################################
    ############################################################################################
    pprint(args)
    return args