#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/../../'))
if rpath not in sys.path:sys.path.append(rpath)

from . trainer import Trainer
from . tester import Tester
from . dataset import Dataset
import argparse
import time,pandas

from ragger.config import args

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=5000, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=1415, type=int, help="batch size")
    parser.add_argument('-save_each', default=50, type=int, help="validate every k epochs")
    SimplE_args = parser.parse_args()
    return SimplE_args

def fit_model():

    SimplE_args = get_parameter()
    dataset = Dataset(args.dataset_name)

    print("~~~~ Training ~~~~")
    trainer = Trainer(dataset, SimplE_args)
    trainer.train()

    dataset = Dataset(args.dataset_name)
    tester = Tester(dataset, f'{fpath}/models/SimplE_{dataset.name}_dim{SimplE_args.emb_dim}.chkpnt', "test")
    measure = tester.test()
    print(measure)

def enhance_data():

    SimplE_args = get_parameter()
    dataset = Dataset(args.dataset_name)
    tester = Tester(dataset, f'{fpath}/models/SimplE_{dataset.name}_dim{SimplE_args.emb_dim}.chkpnt', "train")
    enhance_data=pandas.concat((tester.enhance('head'),tester.enhance('tail')),axis=0,ignore_index=True)

    idToEntity={value:key for key,value in dataset.ent2id.items()}
    idToRelation={value:key for key,value in dataset.rel2id.items()}
    enhance_data['head']=enhance_data['head'].map(lambda one:idToEntity[one])
    enhance_data['relation']=enhance_data['relation'].map(lambda one:idToRelation[one])
    enhance_data['tail']=enhance_data['tail'].map(lambda one:idToEntity[one])

    enhance_data['relation']=enhance_data['relation'].astype('str').str.lstrip('[_/]').str.replace('[/.]','_',regex=True)
    enhance_data['head']=enhance_data['head'].astype('str').str.lstrip('[_/0]').str.replace('[/.]','_',regex=True)
    enhance_data['tail']=enhance_data['tail'].astype('str').str.lstrip('[_/0]').str.replace('[/.]','_',regex=True)

    return enhance_data

def is_model_exist():

    SimplE_args = get_parameter()
    path=f'{fpath}/models/SimplE_{args.dataset_name}_dim{SimplE_args.emb_dim}.chkpnt'
    exist_bool=os.path.exists(path)
    return exist_bool

if __name__ == '__main__':
    pass

    fit_model()

    # SimplE_args = get_parameter()
    # dataset = Dataset(args.dataset_name)

    # print("~~~~ Training ~~~~")
    # trainer = Trainer(dataset, SimplE_args)
    # trainer.train()

    # print("~~~~ Select best epoch on validation set ~~~~")
    # epochs2test = [str(int(SimplE_args.save_each * (i + 1))) for i in range(SimplE_args.ne // SimplE_args.save_each)]
    # dataset = Dataset(args.dataset_name)
    
    # best_mrr = -1.0
    # best_epoch = "0"
    # for epoch in epochs2test:
    #     start = time.time()
    #     print(epoch)
    #     model_path = "models/" + args.dataset_name + "/" + epoch + ".chkpnt"
    #     tester = Tester(dataset, model_path, "valid")
    #     mrr = tester.test()
    #     if mrr > best_mrr:
    #         best_mrr = mrr
    #         best_epoch = epoch
    #     print(time.time() - start)

    # print("Best epoch: " + best_epoch)

    # print("~~~~ Testing on the best epoch ~~~~")
    # best_model_path = "models/" + args.dataset_name + "/" + best_epoch + ".chkpnt"
    # tester = Tester(dataset, best_model_path, "test")
    # tester.test()
