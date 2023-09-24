#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====
os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=400,    help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=10,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=True,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=20211202,     help='Seed for the random number generator');



## Training details
parser.add_argument('--test_interval',  type=int,   default=1,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=50,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="aamsoftmax",     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adamw", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');


## Pre-trained Transformer Model
parser.add_argument('--pretrained_model_path',    type=str,    default="None",  help='Absolute path to the pre-trained model');
parser.add_argument('--weight_finetuning_reg',    type=float,  default=0.001,  help='L2 regularization towards the initial pre-trained model');
parser.add_argument('--LLRD_factor',              type=float,  default=1.0,  help='Layer-wise Learning Rate Decay (LLRD) factor');
parser.add_argument('--LR_Transformer',           type=float,  default=2e-5,  help='Learning rate of pre-trained model');
parser.add_argument('--LR_MHFA',                  type=float,  default=5e-3,  help='Learning rate of back-end attentive pooling model');

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker');
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection');
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and test data
## Training and test data
parser.add_argument('--train_list', type=str, default="/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/train_list.txt", help='Train list');
parser.add_argument('--test_list', type=str, default="/mnt/proj3/open-24-5/pengjy_new/WavLM/vox_list/veri_test.txt", help='Evaluation list');

parser.add_argument('--train_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb2/dev/aac/", help='Absolute path to the train set');
parser.add_argument('--test_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_wav/16kHz/voxceleb_1.1/", help='Absolute path to the test set');
parser.add_argument('--musan_path', type=str, default="/mnt/proj3/open-24-5/pengjy_new/musan_split/", help='Absolute path to the test set');
parser.add_argument('--rir_path', type=str, default="/mnt/proj3/open-24-5/plchot/data_augment/16kHz/simulated_rirs/", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=80,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=192,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="7888", help='Port for distributed training, input as text');
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args();

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## Try to import NSML
try:
    import nsml
    from nsml import HAS_DATASET, DATASET_PATH, PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
    from nsml import NSML_NFS_OUTPUT, SESSION_NAME
except:
    pass;

warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    ## Load models
    s = SpeakerNet(**vars(args));


    s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100];

    # trainLoader = get_data_loader(args.train_list, **vars(args));
    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print("Model {} loaded!".format(args.initial_model));
    elif len(modelfiles) >= 1:
        # print("Model {} loaded from previous state!".format(modelfiles[-2]));
        # trainer.loadParameters(modelfiles[-2]);
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1,it):
        trainer.__scheduler__.step()
    
    
    # pytorch_total_params = sum(p.numel() for p in s.module.__S__.model.feature_extractor.parameters())
    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())


    print('Total parameters: ',pytorch_total_params)
    # quit();
    ## Evaluation code - must run on single GPU
    if args.eval == True:
            scorefile_score  = open(args.result_save_path+"/Eval_scores_mean_O_All.txt", "w");
            print('Test list',args.test_list)

            for i in range(1,5):
                print("Model {} loaded from previous state!".format(modelfiles[-i]));
                trainer.loadParameters(modelfiles[-i]);    
                # trainer.loadParameters(modelfiles[0]);           
       
                # sc, lab, _,sc1,sc2 = trainer.evaluateFromList_1utterance(**vars(args))
                sc, lab, _,sc1,sc2 = trainer.evaluateFromList(**vars(args))

                if args.gpu == 0:

                    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
                    result1 = tuneThresholdfromScore(sc1, lab, [1, 0.1]);
                    result2 = tuneThresholdfromScore(sc2, lab, [1, 0.1]);

                    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)

                    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                    mindcf_1, threshold_1 = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, args.dcf_c_miss, args.dcf_c_fa)

                    print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]),"MinDCF {:2.5f}".format(mindcf));

                    scorefile_score.write("Epoch {}, VEER {:2.4f}, VEER_S1 {:2.4f}, VEER_S2 {:2.4f}, MinDCF05 {:2.5f}, MinDCF01 {:2.5f}\n".format(modelfiles[-i], result[1], result1[1], result2[1], mindcf,mindcf_1));
                    scorefile_score.flush()

            scorefile_score.close()
            return


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    if ("nsml" in sys.modules) and not args.eval:
        args.save_path  = os.path.join(args.save_path,SESSION_NAME.replace('/','_'))

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()
