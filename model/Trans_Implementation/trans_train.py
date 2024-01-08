# -*- coding :utf-8 -*-

import os,json,torch,codecs,pickle,argparse,copy,pandas
import numpy as np
from torch.utils.data import DataLoader
from . trans_config import TransConfig
from . code.utils import utils
from . code.models import TransE, TransH, TransA, TransD, KG2E
from . code.utils import evaluation
from . code.dataloader.dataloader import tripleDataset
from . code.process.triples import csvToStandard, jsonToStandard,generateDict, splitData

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

args = TransConfig()
def transGenerateData():
    
    # Step1: Transform raw data to standard format
    csvToStandard(rawPath=args.trainFile,
                  savePath=args.saveDir+"train.txt",
                  names=["head", "relation", "tail"],
                  header=None,
                  sep="\t",
                  encoding="utf-8")
    csvToStandard(rawPath=args.validFile,
                  savePath=args.saveDir+"valid.txt",
                  names=["head", "relation", "tail"],
                  header=None,
                  sep="\t",
                  encoding="utf-8")
    csvToStandard(rawPath=args.testFile,
                  savePath=args.saveDir+"test.txt",
                  names=["head", "relation", "tail"],
                  header=None,
                  sep="\t",
                  encoding="utf-8")

    # Step2: Generate dict
    generateDict(dataPath=[args.trainFile, args.validFile, args.testFile],
                 dictSaveDir=args.dictDir)

    # Step3: Split data
    pass

def prepareDataloader(args, repSeed, exSeed, headSeed, tailSeed):
    # Initialize dataset and dataloader
    # If print(dataset[:]), you can get the result like:
    #   (np.array(N, 3, dtype=int64), np.array(N, 3, dtype=int64))
    # The first array represents the positive triples, while
    #   the second array represents the negtive ones.
    #   N is the size of all data.
    dataset = tripleDataset(posDataPath=[args.pospath,args.validpath],
                            entityDictPath=args.entpath,
                            relationDictPath=args.relpath)
    dataset.generateNegSamples(repProba=args.repproba,
                               exProba=args.exproba,
                               repSeed=repSeed,
                               exSeed=exSeed,
                               headSeed=headSeed,
                               tailSeed=tailSeed)
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=args.shuffle,
                            num_workers=args.numworkers,
                            drop_last=args.droplast)
    return dataloader

def prepareEvalDataloader(args):
    dataset = tripleDataset(posDataPath=args.testpath,
                            entityDictPath=args.entpath,
                            relationDictPath=args.relpath)
    dataloader = DataLoader(dataset,
                            batch_size=len(dataset),
                            shuffle=False,
                            drop_last=False)
    return dataloader

def prepareEnhanceDataloader(args):
    dataset = tripleDataset(posDataPath=[args.pospath,args.validpath],
                            entityDictPath=args.entpath,
                            relationDictPath=args.relpath)
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize*10,
                            shuffle=False,
                            drop_last=False)
    return dataloader

def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

class trainTriples():
    def __init__(self, args,sumWriter=None):
        self.args = args
        self.sumWriter=sumWriter

    def prepareData(self,is_enhance=False):
        print("INFO : Prepare dataloader")
        # self.dataloader = prepareDataloader(self.args)
        if is_enhance:self.evalloader = prepareEnhanceDataloader(self.args)
        else:self.evalloader = prepareEvalDataloader(self.args)
        self.entityDict = json.load(open(self.args.entpath, "r"))
        self.relationDict = json.load(open(self.args.relpath, "r"))

    def prepareModel(self):
        print("INFO : Init model %s"%self.args.modelname)
        if self.args.modelname == "TransE":
            self.model = TransE.TransE(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       embeddingDim=self.args.TransE["EmbeddingDim"],
                                       margin=self.args.TransE["Margin"],
                                       L=self.args.TransE["L"])
        elif self.args.modelname == "TransH":
            self.model = TransH.TransH(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       embeddingDim=self.args.TransH["EmbeddingDim"],
                                       margin=self.args.TransH["Margin"],
                                       L=self.args.TransH["L"],
                                       C=self.args.TransH["C"],
                                       eps=self.args.TransH["Eps"])
        elif self.args.modelname == "TransA":
            self.model = TransA.TransA(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       embeddingDim=self.args.TransA["EmbeddingDim"],
                                       margin=self.args.TransA["Margin"],
                                       L=self.args.TransA["L"],
                                       lamb=self.args.TransA["Lamb"],
                                       C=self.args.TransA["C"])
        elif self.args.modelname == "TransD":
            self.model = TransD.TransD(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       entityDim=self.args.TransD["EntityDim"],
                                       relationDim=self.args.TransD["RelationDim"],
                                       margin=self.args.TransD["Margin"],
                                       L=self.args.TransD["L"])
        elif self.args.modelname == "KG2E":
            self.model = KG2E.KG2E(entityNum=len(self.entityDict["stoi"]),
                                   relationNum=len(self.relationDict["stoi"]),
                                   embeddingDim=self.args.KG2E["EmbeddingDim"],
                                   margin=self.args.KG2E["Margin"],
                                   sim=self.args.KG2E["Sim"],
                                   vmin=self.args.KG2E["Vmin"],
                                   vmax=self.args.KG2E["Vmax"])
        else:
            print("ERROR : No model named %s"%self.args.modelname)
            exit(1)
        if self.args.usegpu:
            with torch.cuda.device(self.args.gpunum):
                self.model.cuda()

    def loadPretrainEmbedding(self):
        if self.args.modelname == "TransE":
            print("INFO : Loading pre-training entity and relation embedding!")
            self.model.initialWeight(entityEmbedFile=self.args.entityfile,
                                     entityDict=self.entityDict["stoi"],
                                     relationEmbedFile=self.args.relationfile,
                                     relationDict=self.relationDict["stoi"])
        else:
            print("ERROR : Model %s is not supported!"%self.args.modelname)
            exit(1)

    # [TODO]Different models should be considered differently
    def loadPretrainModel(self):
        if self.args.modelname == "TransE":
            print("INFO : Loading pre-training model.")
            modelType = os.path.splitext(self.args.premodel)[-1]
            if modelType == ".param":
                self.model.load_state_dict(torch.load(self.args.premodel))
            elif modelType == ".model":
                self.model = torch.load(self.args.premodel)
            else:
                print("ERROR : Model type %s is not supported!")
                exit(1)
        else:
            print("ERROR : Model %s is not supported!" % self.args.modelname)
            exit(1)

    def loadModel(self):

        print("INFO : Loading model.")
        model_path = os.path.join(self.args.modelpath, "{}_{}_ent{}_rel{}.{}".format(self.args.modelname, self.args.dataset_name, getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"], self.args.modelsave))
        if self.args.modelsave == "param":
            self.model.load_state_dict(torch.load(model_path))
        elif self.args.modelsave == "model":
            self.model = torch.load(model_path)
        else:
            print("ERROR : Model type %s is not supported!")
            exit(1)

    def fit(self):
        EPOCHS = self.args.epochs
        LR = self.args.learningrate
        OPTIMIZER = self.args.optimizer
        if OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         weight_decay=self.args.weightdecay,
                                         lr=LR)
        else:raise ValueError("ERROR : Optimizer %s is not supported."%OPTIMIZER)

        # Training, GLOBALSTEP and GLOBALEPOCH are used for summary
        minLoss = float("inf")
        bestaccuracy = float("inf")
        GLOBALSTEP = 0
        GLOBALEPOCH = 0
        for seed in range(self.args.seeds):
            print("INFO : Using seed %d" % seed)
            self.dataloader = prepareDataloader(self.args, repSeed=seed, exSeed=seed, headSeed=seed, tailSeed=seed)
            for epoch in range(EPOCHS):
                GLOBALEPOCH += 1
                STEP = 0
                print("="*20+"EPOCHS(%d/%d)"%(epoch+1, EPOCHS)+"="*20)
                for posX, negX in self.dataloader:
                    # Allocate tensor to devices
                    if self.args.usegpu:
                        with torch.cuda.device(self.args.gpunum):
                            posX = Variable(torch.LongTensor(posX).cuda())
                            negX = Variable(torch.LongTensor(negX).cuda())
                    else:
                        posX = Variable(torch.LongTensor(posX))
                        negX = Variable(torch.LongTensor(negX))

                    # Normalize the embedding if neccessary
                    self.model.normalizeEmbedding()

                    # Calculate the loss from the model
                    loss = self.model(posX, negX)
                    if self.args.usegpu:
                        lossVal = loss.cpu().item()
                    else:
                        lossVal = loss.item()

                    # Calculate the gradient and step down
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Print infomation and add to summary
                    if minLoss > lossVal:
                        minLoss = lossVal
                    print("[TRAIN-EPOCH(%d/%d)-STEP(%d)]Loss:%.8f, minLoss:%.8f"%(epoch+1, EPOCHS, STEP, lossVal, minLoss))
                    STEP += 1
                    GLOBALSTEP += 1
                    self.sumWriter.add_scalar('train/loss', lossVal, global_step=GLOBALSTEP)
                if GLOBALEPOCH % self.args.lrdecayepoch == 0:
                    adjust_learning_rate(optimizer, decay=self.args.lrdecay)
                if GLOBALEPOCH % self.args.evalepoch == 0:
                    for ranks,_,_ in evaluation.Evaluation(evalloader=self.evalloader,
                                                           model=self.args.modelname,
                                                           simMeasure=args.simmeasure,
                                                           **self.model.retEvalWeights()):pass
                    MR,MRR,Hit1,Hit3,Hit10=evaluation.indicator_Evaluation(ranks)
                    self.sumWriter.add_scalar('train/eval', MR, global_step=GLOBALEPOCH)
                    print(f"[EVALUATION-EPOCH({epoch+1}/{EPOCHS})], eval MR={MR},MRR={MRR},Hit1={Hit1},Hit3={Hit3},Hit10={Hit10},")
                    # Save the model if new accuracy is better
                    if MR < bestaccuracy:
                        bestaccuracy = MR
                        self.saveModel()
                        self.dumpEmbedding()

    def saveModel(self):
        if self.args.modelsave == "param":
            path = os.path.join(self.args.modelpath, "{}_{}_ent{}_rel{}.param".format(self.args.modelname, self.args.dataset_name, getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"]))
            torch.save(self.model.state_dict(), path)
        elif self.args.modelsave == "model":
            path = os.path.join(self.args.modelpath, "{}_{}_ent{}_rel{}.model".format(self.args.modelname, self.args.dataset_name, getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"]))
            torch.save(self.model, path)
        else:
            print("ERROR : Saving mode %s is not supported!"%self.args.modelsave)
            exit(1)

    def dumpEmbedding(self):
        '''
        TXT save type only supports saving embedding and relation embedding
        '''
        if os.path.exists(self.args.embedpath)==False:os.makedirs(self.args.embedpath)
        if self.args.savetype == "txt":
            entWeight = self.model.entityEmbedding.weight.detach().cpu().numpy()
            relWeight = self.model.relationEmbedding.weight.detach().cpu().numpy()
            entityNum, entityDim = entWeight.shape
            relationNum, relationDim = relWeight.shape
            entsave = os.path.join(self.args.embedpath, "entityEmbedding.txt")
            relsave = os.path.join(self.args.embedpath, "relationEmbedding.txt")
            with codecs.open(entsave, "w", encoding="utf-8") as fp:
                fp.write("{} {}\n".format(entityNum, entityDim))
                for ent, embed in zip(self.entityDict["itos"], entWeight):
                    fp.write("{}\t{}\n".format(ent, ",".join(embed.astype(np.str))))
            with codecs.open(relsave, "w", encoding="utf-8") as fp:
                fp.write("{} {}\n".format(relationNum, relationDim))
                for rel, embed in zip(self.relationDict["itos"], relWeight):
                    fp.write("{}\t{}\n".format(rel, ",".join(embed.astype(np.str))))
        elif self.args.savetype == "pkl":
            '''
            pkl saving type dump a dict containing itos list and weights returned by model
            '''
            pklPath = os.path.join(self.args.embedpath, "param_ent{}_rel{}_{}.pkl".format(getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"], self.model))
            with codecs.open(pklPath, "wb") as fp:
                pickle.dump({"entlist" : self.entityDict["itos"],
                             "rellist" : self.relationDict["itos"],
                             "weights" : self.model.retEvalWeights()}, fp)
        else:
            print("ERROR : Format %s is not supported."%self.args.savetype)
            exit(1)

def fit_model():

    sumWriter = SummaryWriter(log_dir=args.summarydir)

    trainModel = trainTriples(args,sumWriter)
    trainModel.prepareData()
    trainModel.prepareModel()
    trainModel.fit()

    sumWriter.close()

def _get_enhance_data(trainModel,idToEntity,idToRelation,eval_type):

    all_enhance_data=pandas.DataFrame([],columns=['head','relation','tail','score',])
    for ranks,simScore,triple in evaluation.Evaluation(evalloader=trainModel.evalloader,
                                                       model=trainModel.args.modelname,
                                                       simMeasure=args.simmeasure,
                                                       eval_type=eval_type,
                                                       **trainModel.model.retEvalWeights()):
        best_entity_index=np.argpartition(simScore,args.candidate_entity_limit)[:,:args.candidate_entity_limit]
        best_entity_score=np.take_along_axis(simScore,best_entity_index,axis=1)
        s_data=pandas.DataFrame({eval_type:best_entity_index.tolist(),'score':best_entity_score.tolist()})
        if eval_type=='tail':r_data=pandas.DataFrame(triple[:,:2],columns=['head','relation'])
        else:r_data=pandas.DataFrame(triple[:,[2,1]],columns=['tail','relation'])

        hrts_data=pandas.concat([r_data,s_data],axis=1)
        enhance_data=hrts_data.explode(eval_type,ignore_index=True)
        enhance_data['score']=hrts_data.explode('score',ignore_index=True)['score']

        enhance_data['head']=enhance_data['head'].map(lambda one:idToEntity[one])
        enhance_data['relation']=enhance_data['relation'].map(lambda one:idToRelation[one])
        enhance_data['tail']=enhance_data['tail'].map(lambda one:idToEntity[one])
        all_enhance_data=pandas.concat([all_enhance_data,enhance_data.loc[:,['head','relation','tail','score']]],axis=0)

    return all_enhance_data

def trans_enhance():

    transGenerateData()

    trainModel = trainTriples(args)
    trainModel.prepareData(is_enhance=True)
    trainModel.loadModel()
    idToEntity={value:key for key,value in trainModel.entityDict['stoi'].items()}
    idToRelation={value:key for key,value in trainModel.relationDict['stoi'].items()}

    enhance_data_tail=_get_enhance_data(trainModel,idToEntity,idToRelation,'tail')
    enhance_data_head=_get_enhance_data(trainModel,idToEntity,idToRelation,'head')
    enhance_data=pandas.concat([enhance_data_tail,enhance_data_head],axis=0,ignore_index=True)

    enhance_data['relation']=enhance_data['relation'].astype('str').str.lstrip('[_/]').str.replace('[/.]','_',regex=True)
    enhance_data['head']=enhance_data['head'].astype('str').str.lstrip('[_/0]').str.replace('[/.]','_',regex=True)
    enhance_data['tail']=enhance_data['tail'].astype('str').str.lstrip('[_/0]').str.replace('[/.]','_',regex=True)

    return enhance_data

if __name__ == "__main__":
    # Print args
    utils.printArgs(args)

    sumWriter = SummaryWriter(log_dir=args.summarydir)
    trainModel = trainTriples(args,sumWriter)
    trainModel.prepareData()
    trainModel.prepareModel()
    if args.loadembed:
        trainModel.loadPretrainEmbedding()
    trainModel.fit()

    sumWriter.close()