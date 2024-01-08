#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/'))
if rpath not in sys.path:sys.path.append(rpath)

import pandas,csv,json
from ragger.config import args

def statisticDataset(dataset_name,is_save=False):
    """
    :Doc->读取数据集中的train.txt、dev.txt、test.txt，返回pandas.DataFrame格式数据
    :Parameters(dataset_name:args.dataset_name_lis中的一个)->需要的数据集名称
    :Parameters(is_save:False or str)->如果为False则返回统计结果，为存储路径则保存为json文件
    :Return(data:pandas.DataFrame)->列为['head','relation','tail']的pandas.DataFrame数据
    """
    data=readDataset(dataset_name)
    train_data=readDataset(dataset_name,'train')
    valid_data=readDataset(dataset_name,'valid')
    test_data=readDataset(dataset_name,'test')
    entity,relation=readDataDict(dataset_name)
    entity,relation=set(entity.values()),set(relation.values())

    train_entity=set(train_data['head'].tolist()+train_data['tail'].tolist())
    valid_entity=set(valid_data['head'].tolist()+valid_data['tail'].tolist())
    test_entity=set(test_data['head'].tolist()+test_data['tail'].tolist())
    train_relation=set(train_data['relation'].drop_duplicates().tolist())
    valid_relation=set(valid_data['relation'].drop_duplicates().tolist())
    test_relation=set(test_data['relation'].drop_duplicates().tolist())

    result=pandas.Series([],dtype='object')
    result.loc['all_triple_num']=len(data)
    result.loc['all_entity_num']=len(entity)
    result.loc['all_relation_num']=len(relation)

    result.loc['train_triple_num']=len(train_data)
    result.loc['train_entity_num']=len(train_entity)
    result.loc['train_relation_num']=len(train_relation)
    result.loc['train_entity_absence_num']=len(entity-train_entity)
    result.loc['train_relation_absence_num']=len(relation-train_relation)
    # result.loc['train_entity_absence']=list(entity-train_entity)
    # result.loc['train_relation_absence']=list(relation-train_relation)

    result.loc['valid_triple_num']=len(valid_data)
    result.loc['valid_entity_num']=len(valid_entity)
    result.loc['valid_relation_num']=len(valid_relation)
    result.loc['valid_entity_absence_num']=len(entity-valid_entity)
    result.loc['valid_relation_absence_num']=len(relation-valid_relation)
    # result.loc['valid_entity_absence']=list(entity-valid_entity)
    # result.loc['valid_relation_absence']=list(relation-valid_relation)

    result.loc['test_triple_num']=len(test_data)
    result.loc['test_entity_num']=len(test_entity)
    result.loc['test_relation_num']=len(test_relation)
    result.loc['test_entity_absence_num']=len(entity-test_entity)
    result.loc['test_relation_absence_num']=len(relation-test_relation)
    # result.loc['test_entity_absence']=list(entity-test_entity)
    # result.loc['test_relation_absence']=list(relation-test_relation)

    result.loc['train_entity_absence_to_valid_num']=len(valid_entity-train_entity)
    result.loc['train_relation_absence_to_valid_num']=len(valid_relation-train_relation)
    result.loc['train_entity_absence_to_test_num']=len(test_entity-train_entity)
    result.loc['train_relation_absence_to_test_num']=len(test_relation-train_relation)
    result.loc['valid_entity_absence_to_test_num']=len(test_entity-valid_entity)
    result.loc['valid_relation_absence_to_test_num']=len(test_relation-valid_relation)

    result.loc['train_triple_same_to_valid_num']=len(train_data.join(valid_data.set_index(['head','relation','tail']),on=['head','relation','tail'],how='inner'))
    result.loc['train_triple_same_to_test_num']=len(train_data.join(test_data.set_index(['head','relation','tail']),on=['head','relation','tail'],how='inner'))
    result.loc['valid_triple_same_to_test_num']=len(valid_data.join(test_data.set_index(['head','relation','tail']),on=['head','relation','tail'],how='inner'))

    result.loc['all_self_link_relation']=data[data['head']==data['tail']]['relation'].drop_duplicates().tolist()
    result.loc['train_self_link_relation']=train_data[train_data['head']==train_data['tail']]['relation'].drop_duplicates().tolist()
    result.loc['valid_self_link_relation']=valid_data[valid_data['head']==valid_data['tail']]['relation'].drop_duplicates().tolist()
    result.loc['test_self_link_relation']=test_data[test_data['head']==test_data['tail']]['relation'].drop_duplicates().tolist()

    result.loc['train_relation_size']={}
    result.loc['valid_relation_size']={}
    result.loc['test_relation_size']={}
    for one in train_relation:
        relation_data=train_data[train_data['relation']==one]
        result.loc['train_relation_size'][one]=len(relation_data)
    for one in valid_relation:
        relation_data=valid_data[valid_data['relation']==one]
        result.loc['valid_relation_size'][one]=len(relation_data)
    for one in test_relation:
        relation_data=test_data[test_data['relation']==one]
        result.loc['test_relation_size'][one]=len(relation_data)

    result.loc['train_relation_undirected_rate']={}
    result.loc['all_relation_undirected_rate']={}
    undirected_data=pandas.concat((train_data,train_data.rename(columns={'tail':'head','head':'tail'})),axis=0)
    undirected_data=undirected_data[undirected_data.duplicated()]
    for one in relation:
        result.loc['train_relation_undirected_rate'][one]=len(undirected_data[undirected_data['relation']==one])/len(train_data[train_data['relation']==one])
    undirected_data=pandas.concat((data,data.rename(columns={'tail':'head','head':'tail'})),axis=0)
    undirected_data=undirected_data[undirected_data.duplicated()]
    for one in relation:
        result.loc['all_relation_undirected_rate'][one]=len(undirected_data[undirected_data['relation']==one])/len(data[data['relation']==one])
    
    # result.loc['train_relation_head_functionality']={}
    # result.loc['train_relation_tail_functionality']={}
    # for one in train_relation:
    #     relation_data=train_data[train_data['relation']==one]
    #     result.loc['train_relation_head_functionality'][one]=len(relation_data['head'].drop_duplicates())/len(relation_data)
    #     result.loc['train_relation_tail_functionality'][one]=len(relation_data['tail'].drop_duplicates())/len(relation_data)

    if is_save==False:return result
    elif is_save==True:result.to_json('./test.json',indent=4)
    elif isinstance(is_save,str):result.to_json(is_save,indent=4)
    else:raise ValueError('is_save=%s输入错误'%is_save)

def readDataset(dataset_name,data_name=None):
    """
    :Doc->读取数据集中的train.txt、dev.txt、test.txt，返回pandas.DataFrame格式数据
    :Parameters(dataset_name:args.dataset_name_lis中的一个)->需要的数据集名称
    :Parameters(data_name:'train' or 'valid' or 'test' or None)->如果为None则返回三个数据的合并，其他则返回对应数据
    :Return(data:pandas.DataFrame)->列为['head','relation','tail']的pandas.DataFrame数据
    """
    # 得到不同数据集的路径
    if dataset_name in args.dataset_name_lis:dataset_path='%s/%s'%(args.dataset_path,dataset_name)
    else:raise ValueError('dataset_name输入错误')

    # 如果data_name为None，则将三个文件进行拼接
    if data_name==None:
        data=pandas.DataFrame([])
        for one in ['/train.txt','/valid.txt','/test.txt',]:
            temp=pandas.read_csv(dataset_path+one,sep='	',names=['head','relation','tail'],encoding=args.encode)
            data=pandas.concat((data,temp),axis=0)
    # 如果data_name不为None，则只取传入的文件名称对应的文件
    else:
        assert (data_name=='train') or (data_name=='valid') or (data_name=='test') or ('enhance' in data_name),'data_name输入错误'
        data=pandas.read_csv('%s/%s.txt'%(dataset_path,data_name),sep='	',names=['head','relation','tail'],encoding=args.encode)

    # 如果对数据进行去重并将格式变为prolog与ASP可接受的形式
    data.drop_duplicates(inplace=True,ignore_index=True)
    data['relation']=data['relation'].astype('str').str.lstrip('[_/]').str.replace('[/.]','_',regex=True)
    data['head']=data['head'].astype('str').str.lstrip('[_/]').str.replace('[/.]','_',regex=True)
    data['tail']=data['tail'].astype('str').str.lstrip('[_/]').str.replace('[/.]','_',regex=True)
    return data

def readDataDict(dataset_name):
    """
    :Doc->读取数据集中的entity.txt,relation.txt，返回dict格式数据
    :Parameters(dataset_name:args.dataset_name_lis中的一个)->需要的数据集名称
    :Return(data:dict)->数据集中id到所有实体与关系的映射字典
    """
    # 得到不同数据集的路径
    if dataset_name in args.dataset_name_lis:dataset_path='%s/%s'%(args.dataset_path,dataset_name)
    else:raise ValueError('dataset_name输入错误')

    # 读取的实体与关系都作为str格式，并转化为字典
    entity=pandas.read_csv(dataset_path+'/entity.txt',sep='	',names=['index','entity'],encoding=args.encode).astype('str').to_dict()['entity']
    relation=pandas.read_csv(dataset_path+'/relation.txt',sep='	',names=['index','relation'],encoding=args.encode).astype('str').to_dict()['relation']
    return entity,relation

def readDataStatistic(dataset_name):
    """
    :Doc->读取数据集中的statistic.json，返回字典格式
    :Parameters(dataset_name:args.dataset_name_lis中的一个)->需要的数据集名称
    :Return(data:dict)->数据集统计字典
    """
    # 得到不同数据集的路径
    if dataset_name in args.dataset_name_lis:dataset_path='%s/%s'%(args.dataset_path,dataset_name)
    else:raise ValueError('dataset_name输入错误')

    # 读取的实体与关系都作为str格式，并转化为字典
    statistic=pandas.read_json(dataset_path+'/statistic.json',typ='series',encoding=args.encode)
    return statistic

def generateDataDict(renew=False):
    """
    :Doc->得到所有数据集的实体字典与关系字典，以entity.txt与relation.txt存储
    :Parameters(renew:bool)->如果为真，则必定更新所有数据集的entity.txt与relation.txt
    :Return(None)->
    """
    print('查看是否需要获取数据集的实体与关系字典...')
    for dataset_name in args.dataset_name_lis:
        path='%s/%s/'%(args.dataset_path,dataset_name)
        if renew==True or os.path.exists(path+'entity.txt')==False:
            print('获取%s数据集的实体与关系字典...'%dataset_name)
            data=readDataset(dataset_name)
            entity=pandas.concat((data['head'],data['tail']),axis=0).drop_duplicates().reset_index(drop=True)
            relation=data['relation'].drop_duplicates().reset_index(drop=True)
            entity.to_csv(path+'entity.txt',index=True,header=False,quoting=csv.QUOTE_NONE,sep='	',escapechar='',encoding=args.encode)
            relation.to_csv(path+'relation.txt',index=True,header=False,quoting=csv.QUOTE_NONE,sep='	',escapechar='',encoding=args.encode)

def generateDataStatistic(renew=False):
    """
    :Doc->得到所有数据集的实体字典与关系字典，以entity.txt与relation.txt存储
    :Parameters(renew:bool)->如果为真，则必定更新所有数据集的entity.txt与relation.txt
    :Return(None)->
    """
    print('查看是否需要生成数据集的统计...')
    for dataset_name in args.dataset_name_lis:
        path='%s/%s/'%(args.dataset_path,dataset_name)
        if renew==True or os.path.exists(path+'statistic.json')==False:
            print('获取%s数据集的统计...'%dataset_name)
            statisticDataset(dataset_name,path+'statistic.json')

class DataManage(object):

    def __init__(self,dataset_name):
        """
        :Doc->管理运行的数据，生成需要的数据
        :Parameters(dataset_name:args.dataset_name_lis中的一个)->使用的数据集名称
        :Return(None)->
        """
        assert dataset_name in args.dataset_name_lis,'dataset_name输入错误'
        self.dataset_name=dataset_name
        self.data=readDataset(dataset_name,'train').sort_values(by=['relation'],ignore_index=True)
        self.valid_data=readDataset(dataset_name,'valid').sort_values(by=['relation'],ignore_index=True)
        self.test_data=readDataset(dataset_name,'test').sort_values(by=['relation'],ignore_index=True)

        self.idToEntity,self.idToRelation=readDataDict(dataset_name)
        self.data_statistic=readDataStatistic(dataset_name)

    def generateData(self,relation):

        bk_data=self.data
        pos_data=self.data[self.data['relation']==relation]
        gnl_data=self.valid_data[self.valid_data['relation']==relation]
        
        return bk_data,pos_data,gnl_data
    
    def get_enhance_data(self):
        
        return readDataset(self.dataset_name,f'enhance_{args.embed_model}').sort_values(by=['relation'],ignore_index=True)

if __name__=='__main__':

    dataset_name='WN18'
    # data=readDataset(dataset_name,data_name=None)
    # data=readDataDict(dataset_name)
    # data=readDataStatistic(dataset_name)
    # data=readILPSettingData()
    # generateDataDict(renew=True)
    statisticDataset(dataset_name,is_save='./test.json')
    # generateDataStatistic(renew=True)