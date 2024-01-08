#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/../../'))
if rpath not in sys.path:sys.path.append(rpath)

import torch
from . code.utils.utils import CheckPath
from ragger.config import args

class TransConfig():
    def __init__(self):

        # Generate arguments
        self.dataset_name=args.dataset_name
        self.trainFile = f'{args.dataset_path}/{args.dataset_name}/train.txt'
        self.validFile = f'{args.dataset_path}/{args.dataset_name}/valid.txt'
        self.testFile = f'{args.dataset_path}/{args.dataset_name}/test.txt'
        self.saveDir = f'{args.model_path}/Trans_Implementation/data/'
        self.dictDir = f'{args.model_path}/Trans_Implementation/source/dict/'

        # Data arguments
        self.pospath = self.saveDir+"train.txt"
        self.validpath = self.saveDir+"valid.txt"
        self.testpath = self.saveDir+"test.txt"
        self.entpath = self.dictDir+"entityDict.json"
        self.relpath = self.dictDir+"relationDict.json"
        self.embedpath = f'{args.model_path}/Trans_Implementation/source/embed/{args.embed_model}_{args.dataset_name}/'
        self.logpath = f'{args.model_path}/Trans_Implementation/source/log/'
        self.savetype = "txt"

        # Dataloader arguments
        self.batchsize = 1024
        self.shuffle = True
        self.numworkers = 4
        self.droplast = False
        self.repproba = 0.5
        self.exproba = 0.5

        # Model and training general arguments
        self.TransE = {"EmbeddingDim": 100,
                       "Margin":       1.0,
                       "L":            2}
        self.TransH = {"EmbeddingDim": 100,
                       "Margin":       1.0,
                       "L":            2,
                       "C":            0.01,
                       "Eps":          0.001}
        self.TransD = {"EntityDim":    100,
                       "RelationDim":  100,
                       "Margin":       2.0,
                       "L":            2}
        self.TransA = {"EmbeddingDim": 100,
                       "Margin":       3.2,
                       "L":            2,
                       "Lamb":         0.01,
                       "C":            0.2}
        self.KG2E   = {"EmbeddingDim": 100,
                       "Margin":       4.0,
                       "Sim":          "EL",
                       "Vmin":         0.03,
                       "Vmax":         3.0}
        self.usegpu = torch.cuda.is_available()
        self.gpunum = 0
        self.modelname = args.embed_model
        self.weightdecay = 0

        self.seeds=100
        self.epochs = 6
        self.evalepoch = 1
        self.lrdecayepoch = 1

        self.learningrate = 0.01
        self.lrdecay = 0.96
        self.optimizer = "Adam"
        self.simmeasure = "L2"
        self.modelsave = "model"
        self.modelpath = f'{args.model_path}/Trans_Implementation/source/model/'

        self.loadembed = False
        self.entityfile = self.embedpath+"entityEmbedding.txt"
        self.relationfile = self.embedpath+"relationEmbedding.txt"
        self.premodel = self.modelpath+"TransE_WN18_ent100_rel100.model"

        # Other arguments
        self.summarydir = f'{args.model_path}/Trans_Implementation/source/summary/'

        # 嵌入预增强设置
        self.candidate_entity_limit=args.candidate_entity_limit

        # Check Path
        self.CheckPath()

        # self.usePaperConfig()

    def usePaperConfig(self):
        # Paper best params
        if self.modelname == "TransE":
            self.embeddingdim = 50
            self.learningrate = 0.01
            self.margin = 1.0
            self.distance = 1
            self.simmeasure = "L1"
        elif self.modelname == "TransH":
            self.batchsize = 1200
            self.embeddingdim = 50
            self.learningrate = 0.005
            self.margin = 0.5
            self.C = 0.015625
        elif self.modelname == "TransD":
            self.batchsize = 4800
            self.entitydim = 100
            self.relationdim = 100
            self.margin = 2.0

    def CheckPath(self):

        # Check dirs
        CheckPath(self.modelpath, raise_error=False)
        CheckPath(self.summarydir, raise_error=False)
        CheckPath(self.logpath, raise_error=False)
        CheckPath(self.embedpath, raise_error=False)


