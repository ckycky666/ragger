#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/../../'))
if rpath not in sys.path:sys.path.append(rpath)

import torch,tqdm,pandas
import numpy as np
from . measure import Measure

from torch.utils.data import Dataset as TDataset,DataLoader

from ragger.config import args

class EnhanceDataset(TDataset):
    def __init__(self, dataset,data_type,enhance_side='tail'):

        self.dataset = dataset
        self.data_type=data_type
        self.enhance_side=enhance_side
    
    def __len__(self):
        return len(self.dataset.data[self.data_type])
    
    def __getitem__(self, index):
        # 返回第index个数据样本
        head, rel, tail = self.dataset.data[self.data_type][index]
        rels=torch.LongTensor([rel]*self.dataset.num_ent())
        if  self.enhance_side== "head":
            heads=torch.arange(0,self.dataset.num_ent(),dtype=torch.int64)
            tails=torch.LongTensor([tail]*self.dataset.num_ent())
        elif self.enhance_side == "tail":
            heads=torch.LongTensor([head]*self.dataset.num_ent())
            tails=torch.arange(0,self.dataset.num_ent(),dtype=torch.int64)
        return  heads,rels,tails


class Tester:
    def __init__(self, dataset, model_path, data_type):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if type(model_path)==str:self.model = torch.load(model_path, map_location = self.device)
        else:self.model=model_path
        self.model.eval()
        self.dataset = dataset
        self.data_type = data_type #["train", "valid", "test"]
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

    def get_rank(self, sim_scores,real):
        return (sim_scores >= sim_scores[int(real)]).sum()

    def create_queries(self, fact, head_or_tail):
        head, rel, tail = fact
        rels=[rel]*self.dataset.num_ent()
        if head_or_tail == "head":
            heads=np.arange(0,self.dataset.num_ent())
            tails=[tail]*self.dataset.num_ent()
            real=head
        elif head_or_tail == "tail":
            heads=[head]*self.dataset.num_ent()
            tails=np.arange(0,self.dataset.num_ent())
            real=tail
        return torch.LongTensor(heads).to(self.device), torch.LongTensor(rels).to(self.device), torch.LongTensor(tails).to(self.device),real
    
    def test(self,test_side='tail'):

        measure = Measure()
        for i, fact in tqdm.tqdm(list(enumerate(self.dataset.data[self.data_type])),desc='测试进度'):
            h, r, t, real = self.create_queries(fact, test_side)
            sim_scores = self.model(h, r, t).cpu().data.numpy()
            rank = self.get_rank(sim_scores,real)
            measure.update(rank)

        measure.normalize(len(self.dataset.data[self.data_type]))
        return measure
    
    # def enhance(self,enhance_side='tail'):

    #     enhance_data=pandas.DataFrame([],columns=['head','relation','tail','score'])
    #     for i, fact in tqdm.tqdm(list(enumerate(self.dataset.data[self.data_type])),desc='测试进度'):
    #         h, r, t, real = self.create_queries(fact, enhance_side)
    #         sim_scores = self.model(h, r, t).cpu().data.numpy()
    #         temp_data=pandas.DataFrame({'head':h.cpu(),'relation':r.cpu(),'tail':t.cpu(),'score':sim_scores})
    #         temp_data=temp_data.nlargest(args.candidate_entity_limit,'score')
    #         enhance_data=pandas.concat((enhance_data,temp_data),axis=0)
    #     return enhance_data
    
    def enhance(self,enhance_side='tail'):

        enhance_dataset=EnhanceDataset(self.dataset,self.data_type,enhance_side)
        enhance_dataloader=DataLoader(enhance_dataset,batch_size=1,num_workers=4)
        enhance_data=pandas.DataFrame([],columns=['head','relation','tail','score'])
        for h, r, t in tqdm.tqdm(enhance_dataloader,desc='测试进度'):
            h,r,t=h[0], r[0], t[0]
            sim_scores = self.model(h.to(self.device),r.to(self.device),t.to(self.device)).cpu().data.numpy()
            temp_data=pandas.DataFrame({'head':h,'relation':r,'tail':t,'score':sim_scores})
            temp_data=temp_data.nlargest(args.candidate_entity_limit,'score')
            enhance_data=pandas.concat((enhance_data,temp_data),axis=0)
        return enhance_data

    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data[spl]:
                tuples.append(tuple(fact))
        
        return tuples



    
    
