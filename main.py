#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/'))
if rpath not in sys.path:sys.path.append(rpath)

import pandas,csv,profile,pstats,tqdm,time,re,copy,threading,pickle
from pyswip import Prolog
from pyswip.prolog import PrologError
from concurrent.futures import ThreadPoolExecutor

from ragger.config import args
from ragger.utils import DataManage
from ragger.utils import generateDataDict,generateDataStatistic
from ragger.model.ckyrule.loop import CKYRule
from ragger.model.ckyrule.core import Program,Rule,Literal

from ragger.model.Trans_Implementation import trans_train
from ragger.model.SimplE import simple_main
from ragger.model.InteractE import interacte

class embedManage(object):

    def __init__(self,data_manage,fact_data):

        self.data_manage=data_manage
        self.fact_data=fact_data
        if args.embed_model in ['TransE','TransH','TransA','TransD','KG2E']:self.embed_program='Trans_Implementation'
        elif args.embed_model in ['SimplE','InteractE']:self.embed_program=args.embed_model
        else:raise ValueError('args.embed_model=%s设置错误'%args.embed_model)

    def _prepareData(self):

        if self.embed_program=='Trans_Implementation':trans_train.transGenerateData()
        elif self.embed_program=='SimplE':pass
        elif self.embed_program=='InteractE':pass
        else:raise ValueError('self.embed_program=%s设置错误'%self.embed_program)

    def _fit_model(self):

        if self.embed_program=='Trans_Implementation':trans_train.fit_model()
        elif self.embed_program=='SimplE':simple_main.fit_model()
        elif self.embed_program=='InteractE':interacte.fire()
        else:raise ValueError('self.embed_program=%s设置错误'%self.embed_program)

    def _enhance_data(self):

        if self.embed_program=='Trans_Implementation':original_enhance_data=trans_train.trans_enhance()
        elif self.embed_program=='SimplE':original_enhance_data=simple_main.enhance_data()
        elif self.embed_program=='InteractE':original_enhance_data=interacte.enhance_data()
        else:raise ValueError('self.embed_program=%s设置错误'%self.embed_program)
        #去除增强数据中，与训练集完全相同的三元组
        temp=pandas.concat((original_enhance_data,self.fact_data),ignore_index=True)
        original_enhance_data=temp[~temp.duplicated(keep='last',subset=['head','relation','tail'])].loc[:original_enhance_data.shape[0]-1,:].reset_index(drop=True)
        #按照关系划分，每个关系获得的增强数量相等，在每个关系中按照实体预测评分排序，取前relation_take_num个
        enhance_data=pandas.DataFrame([])
        relation_take_num=int(self.fact_data.shape[0]*args.enhance_data_scale/self.data_manage.data_statistic['test_relation_num'])
        for relation in self.data_manage.data_statistic['test_relation_size'].keys():
            relation_data=original_enhance_data[original_enhance_data['relation']==relation]
            relation_data=relation_data.sort_values(by='score',ignore_index=True).head(relation_take_num).loc[:,['head','relation','tail']]
            enhance_data=pandas.concat([enhance_data,relation_data],axis=0,ignore_index=True)
        enhance_data.to_csv(f'{args.dataset_path}/{args.dataset_name}/enhance_{args.embed_model}.txt',index=False,header=False,sep='	',encoding=args.encode)

    def is_model_exist(self):

        if self.embed_program=='Trans_Implementation':
            path=f'{args.model_path}/Trans_Implementation/source/embed/{args.embed_model}_{args.dataset_name}/relationEmbedding.txt'
            exist_bool=os.path.exists(path)
        elif self.embed_program=='SimplE':exist_bool=simple_main.is_model_exist()
        elif self.embed_program=='InteractE':exist_bool=interacte.is_model_exist()
        else:raise ValueError('self.embed_program=%s设置错误'%self.embed_program)
        return exist_bool

    def is_enhance_exist(self):

        path=f'{args.dataset_path}/{args.dataset_name}/enhance_{args.embed_model}.txt'
        return os.path.exists(path)

    def train(self):

        self._prepareData()
        self._fit_model()

    def enhance(self):

        self._enhance_data()

class Ragger(object):

    def __init__(self,program_name,new_model=False,new_enhance=False):

        if not os.path.exists(f'{args.model_path}/Trans_Implementation/data'):os.makedirs(f'{args.model_path}/Trans_Implementation/data')
        if not os.path.exists(f'{args.model_path}/Trans_Implementation/source/dict'):os.makedirs(f'{args.model_path}/Trans_Implementation/source/dict')
        
        generateDataDict(renew=False)
        generateDataStatistic(renew=False)

        #设置基本参数
        self.program_name=program_name
        self.data_manage=DataManage(args.dataset_name)
        self.path=f'{args.program_path}/{self.program_name}'
        self.search=CKYRule(data_manage=self.data_manage,
                            is_debug=args.is_debug,
                            is_sequential_covering=args.is_sequential_covering,
                            is_specialize_good=args.is_specialize_good,
                            is_diff_varcons=args.is_diff_varcons,)
        #得到规则条件谓词
        self.search_predicate_lis=list(self.data_manage.data_statistic['test_relation_size'].keys())
        
        #得到事实库
        self.pre_fact_data=pandas.concat((self.data_manage.data,self.data_manage.valid_data),axis=0).sort_values(by=['relation'],ignore_index=True)
        if args.is_enhance_dataset:
            embed=embedManage(self.data_manage,self.pre_fact_data)
            print('查看是否需要训练嵌入模型...')
            if embed.is_model_exist()==False or new_model==True:embed.train()
            print('查看是否需要进行数据增强...')
            if embed.is_enhance_exist()==False or new_enhance==True:embed.enhance()
            self.enhance_data=self.data_manage.get_enhance_data()

            relation_data_lis=[]
            for relation in tqdm.tqdm(list(self.data_manage.data_statistic['train_relation_size'].keys()),desc='将增强的数据按序融入原数据'):
                relation_data_lis.append(self.pre_fact_data[self.pre_fact_data['relation']==relation])

                enhance_relation_data=self.enhance_data[self.enhance_data['relation']==relation]
                relation_data_lis.append(enhance_relation_data)
            self.fact_data=pandas.concat(relation_data_lis,axis=0)
        else:
            self.fact_data=self.pre_fact_data
            self.enhance_data=None

        #保存中间结果
        self._temp_program_prolog=None
        self._temp_program_clingo=None

    def searchLogicProgram(self,search_lis=[]):

        final_program=Program(self.program_name)
        if search_lis==[]:search_lis=self.search_predicate_lis
        print('---------开始规则挖掘---------')
        for head_predicate in tqdm.tqdm(search_lis,desc='搜索总体进度'):
            # head_predicate='synset_domain_topic_of'

            print('搜索关于%s的规则...'%head_predicate)
            program=self.search.search(head_predicate)
            final_program.add_clause(program.clause)
            print('得到关于谓词%s的逻辑程序如下'%head_predicate)
            print(program)

            #保存中间结果，避免搜索中间出问题导致结果全部丢失
            self._temp_program_clingo=pandas.concat((self._temp_program_clingo,program.to_clingo('',is_contain_fact=False,is_save=False)),axis=0).reset_index(drop=True)
            self._temp_program_clingo.to_csv(f'{self.path}/Program.lp',sep='|',index=False,header=False,encoding=args.encode)

            # break
        return final_program

    def main(self,continue_mode=False):

        if os.path.exists(self.path)==False:os.makedirs(self.path)
        #保存参数设置
        with open('./config.py',mode='r',encoding=args.encode) as f:
            param=f.readlines()
        with open(f'{self.path}/param.py',mode='w',encoding=args.encode) as f:
            f.writelines(param)
        #如果是追加模式，则得到未完成的search_lis
        if continue_mode:
            try:
                self._temp_program_clingo=pandas.read_csv(f'{self.path}/Program.lp',sep='|',names=['clingo'],header=None,encoding=args.encode)['clingo']
                search_lis=set(self.search_predicate_lis)-set(self._temp_program_clingo.str.extract('(.*)\(.*\):-.*',expand=False).drop_duplicates())
            except FileNotFoundError:
                self._temp_program_clingo=pandas.Series([],dtype=object)
                search_lis=[]
        else:
            self._temp_program_clingo=pandas.Series([],dtype=object)
            search_lis=[]

        final_program=self.searchLogicProgram(search_lis)

    def _test_func1(self,line,hit_dict,prolog,enhance_set,relation_data_series,is_reverse_side=False):

        if is_reverse_side:test_side='head'
        else:test_side='tail'

        if 'Hit@1_'+line['relation'] not in hit_dict:hit_dict['Hit@1_'+line['relation']]=0
        if 'Hit@3_'+line['relation'] not in hit_dict:hit_dict['Hit@3_'+line['relation']]=0
        if 'Hit@10_'+line['relation'] not in hit_dict:hit_dict['Hit@10_'+line['relation']]=0
        if 'Hit@n_'+line['relation'] not in hit_dict:hit_dict['Hit@n_'+line['relation']]=0
        if 'MRR_'+line['relation'] not in hit_dict:hit_dict['MRR_'+line['relation']]=0
        if 'MEN_'+line['relation'] not in hit_dict:hit_dict['MEN_'+line['relation']]=0

        try:
            if is_reverse_side:result=list(prolog.query(f"get_possible_conditions(X,{line['relation']}_r(X,{line['tail']}),Y)."))[0]['Y']
            else:result=list(prolog.query(f"get_possible_conditions(X,{line['relation']}_r({line['head']},X),Y)."))[0]['Y']
            condition=[re.findall('[a-z_0-9]+\([a-z_0-9]+, [a-z_0-9]+\)',one) for one in result]
            result=[re.search('.*, ([a-z_0-9]+)\)',one)[1] for one in result]
        #由于存在有的查询涉及规则太多，递归超过限度，所以还是跳过
        except (PrologError, TypeError, OSError) as e:
            print(e)
            return

        # print(line)
        # print(result)

        #将因为数据增强部分多推理出的结果放到排序后面，这样一来，数据增强后比数据增强前的准确指标只高不低
        if args.is_enhance_dataset:
            extra_result_index=[]
            for i,con in enumerate(condition):
                if any([one in enhance_set for one in con]):
                    extra_result_index.append(i)
            result=pandas.Series(result,dtype=object)
            move_result=result.loc[extra_result_index]
            result=result.drop(extra_result_index).append(move_result).drop_duplicates().tolist()
        else:result=pandas.Series(result,dtype=object).drop_duplicates().tolist()

        #将候选实体列表中已经存在于事实库的三元组排除出列表
        if is_reverse_side:result_triple=pandas.concat((pandas.Series(result,name='head',dtype=object),pandas.Series([line['relation']]*len(result),name='relation',dtype=object),pandas.Series([line['tail']]*len(result),name='tail',dtype=object)),axis=1)
        else:result_triple=pandas.concat((pandas.Series([line['head']]*len(result),name='head',dtype=object),pandas.Series([line['relation']]*len(result),name='relation',dtype=object),pandas.Series(result,name='tail',dtype=object)),axis=1)
        try:relation_data=relation_data_series.loc[line['relation']]
        except KeyError:
            relation_data=self.pre_fact_data[self.pre_fact_data['relation']==line['relation']]
            relation_data_series.loc[line['relation']]=relation_data
        temp=pandas.concat((result_triple,relation_data),ignore_index=True)
        result_triple=temp[~temp.duplicated(keep='last')].loc[:result_triple.shape[0]-1,:].reset_index(drop=True)
        result=result_triple[test_side].tolist()

        # print(result)
        # input()

        if line[test_side] in result[:1]:
            hit_dict['Hit@1_all']+=1
            hit_dict['Hit@1_'+line['relation']]+=1
        if line[test_side] in result[:3]:
            hit_dict['Hit@3_all']+=1
            hit_dict['Hit@3_'+line['relation']]+=1
        if line[test_side] in result[:10]:
            hit_dict['Hit@10_all']+=1
            hit_dict['Hit@10_'+line['relation']]+=1
        if line[test_side] in result[:]:
            hit_dict['Hit@n_all']+=1
            hit_dict['Hit@n_'+line['relation']]+=1
            hit_dict['MRR_all']+=1/(result.index(line[test_side])+1)
            hit_dict['MRR_'+line['relation']]+=1/(result.index(line[test_side])+1)

        hit_dict['MEN_all']+=len(result)
        hit_dict['MEN_'+line['relation']]+=len(result)

    def _test_func2(self,hit_dict,hit_str):

        if hit_str in ['Hit@1','Hit@3','Hit@n']:start_index=6
        elif hit_str=='Hit@10':start_index=7
        elif hit_str=='MRR' or hit_str=='MEN':start_index=4
        else:raise ValueError('hit_str输入错误')
        print(f'指标{hit_str}统计情况如下:')
        Hit_indicator=pandas.Series([hit_dict[f'{hit_str}_all']/len(self.data_manage.test_data)],index=['all'],name=hit_str)
        for key,value in hit_dict.items():
            if f'{hit_str}_' in key and key!=f'{hit_str}_all':Hit_indicator[key[start_index:]]=value/self.data_manage.data_statistic['test_relation_size'][key[start_index:]]
        print(Hit_indicator,'\n')
        return Hit_indicator

    def test(self,revamp_lis=[]):

        #根据Program.lp文件动态生成KnowledgeBase.pl文件，因为self.fact_data内容会随着是否增强数据集改变
        program_clingo=pandas.read_csv(f'{self.path}/Program.lp',sep='|',names=['clingo'],header=None,encoding=args.encode)['clingo']
        program_clingo=program_clingo.str.replace('!=','\\=')
        program_clingo=program_clingo.str.replace('(.*)(\(.*\)):-',lambda p:f'{p.group(1)}_r{p.group(2)}:-',regex=True)
        program_clingo.loc[len(program_clingo)]='get_possible_conditions(Result,Goal,Constraints):- findall(Constr,(clause(Goal,B),call(B),unify_with_occurs_check(Result,X),Constr=(B,X)),Cs), list_to_set(Cs, Constraints).'

        knowledge_base=self.fact_data.apply(lambda line:Rule(head=Literal(line['relation'],(line['head'],line['tail']))),axis=1)
        knowledge_base=Program(self.program_name,knowledge_base)
        knowledge_base=pandas.concat((knowledge_base.to_prolog(f'{self.path}/KnowledgeBase.pl',is_contain_fact=True,is_difference_head=True,is_save=False),program_clingo),axis=0).reset_index(drop=True)
        knowledge_base.to_csv(f'{self.path}/KnowledgeBase.pl',sep='|',index=False,header=False,encoding=args.encode)

        prolog = Prolog()
        if revamp_lis:prolog.consult(f'{self.path}/RevampBase.pl'.replace('\\','/'))
        else:prolog.consult(f'{self.path}/KnowledgeBase.pl'.replace('\\','/'))

        if os.path.exists('./temp_test.pickle'):
            with open('./temp_test.pickle', 'rb') as file:
                prepare_test_i,hit_dict,relation_data_series,already_test_head,enhance_set = pickle.load(file)
        else:
            prepare_test_i=0
            hit_dict={'Hit@1_all':0,'Hit@3_all':0,'Hit@10_all':0,'Hit@n_all':0,'MRR_all':0,'MEN_all':0}
            relation_data_series=pandas.Series([],dtype=object)
            already_test_head=set([])
            #用于与condition里的实例化谓词比较，若condition里存在enhance_set里的内容，则对应的规则结论需要放到最后
            if args.is_enhance_dataset:enhance_set=set((self.enhance_data['relation']+'('+self.enhance_data['head']+', '+self.enhance_data['tail']+')').tolist())
            else:enhance_set=None

        for i in tqdm.tqdm(range(len(self.data_manage.test_data)),desc='规则模型测试进度'):
            #根据规则排序得到候选实体列表
            line=self.data_manage.test_data.iloc[i,:]
            #如果进行revamp，那么跳过不在revamp_lis中的关系
            if revamp_lis and line['relation'] not in revamp_lis:continue

            #每测试一定数量保存一次测试进度，再次加载的时候跳过已经测试的三元组
            if i<prepare_test_i:continue
            if i%10==0:
                with open('./temp_test.pickle', 'wb') as file:
                    pickle.dump([i,hit_dict,relation_data_series,already_test_head,enhance_set], file)

            #默认情况下进行尾查询，当遇到第二个头实体与关系相同的三元组时，进行头查询
            hr_str=line['head']+line['relation']
            if hr_str in already_test_head:
                self._test_func1(line,hit_dict,prolog,enhance_set,relation_data_series,is_reverse_side=True)
            else:
                self._test_func1(line,hit_dict,prolog,enhance_set,relation_data_series,is_reverse_side=False)
                already_test_head.add(hr_str)

        Hit_1=self._test_func2(hit_dict,'Hit@1')
        Hit_3=self._test_func2(hit_dict,'Hit@3')
        Hit_10=self._test_func2(hit_dict,'Hit@10')
        Hit_n=self._test_func2(hit_dict,'Hit@n')
        MRR=self._test_func2(hit_dict,'MRR')
        MEN=self._test_func2(hit_dict,'MEN')
        evaluate_indicator=pandas.concat((Hit_1,Hit_3,Hit_10,Hit_n,MRR,MEN),axis=1)

        if os.path.exists('./temp_test.pickle'):os.remove('./temp_test.pickle')
        if revamp_lis:return evaluate_indicator
        else:evaluate_indicator.to_csv(f'{args.program_path}/{self.program_name}/Accuracy.csv')

    def revamp(self,indicator,bad_rule_limit):
    #由于链路预测任务中规则模型是离散的，对于准确率较低的规则可以通过设置不同的超参数得到优化，并更新到原先的规则模型中

        #根据传入的参数确定需要进行搜索的关系，并进行搜索
        accuracy_data=pandas.read_csv(f'{self.path}/Accuracy.csv',index_col=0)
        accuracy_data=accuracy_data[~(accuracy_data.index=='all')]
        search_lis=list(accuracy_data[accuracy_data[indicator]<=bad_rule_limit].index)
        final_program=self.searchLogicProgram(search_lis)

        #将搜索得到的逻辑程序进行测试，得到测试结果
        final_data=self.fact_data.apply(lambda line:Rule(head=Literal(line['relation'],(line['head'],line['tail']))),axis=1)
        revamp_base=Program(self.program_name,final_data)
        revamp_base.add_clause(final_program.clause)
        revamp_base.to_prolog(f'{self.path}/RevampBase.pl',is_contain_fact=True,is_difference_head=True,is_save=True)
        with open(f'{self.path}/RevampBase.pl','a') as f:
            f.write('get_possible_conditions(Result,Goal,Constraints):- findall(Constr,(clause(Goal,B),call(B),unify_with_occurs_check(Result,X),Constr=(B,X)),Cs), list_to_set(Cs, Constraints).')
        evaluate_indicator=self.test(revamp_lis=search_lis)
        evaluate_indicator=evaluate_indicator[~(evaluate_indicator.index=='all')]

        #根据测试结果与原先的关系准确率进行比较，若MRR得到了改进，则认为需要更新原先的逻辑程序
        flow_data=pandas.DataFrame([])
        for column in evaluate_indicator:
            column_data=evaluate_indicator.loc[:,column]
            flow_indicator=(column_data-accuracy_data.loc[column_data.index,column]).rename(f'{column_data.name}_flow')
            flow_data=pandas.concat((flow_data,column_data,flow_indicator),axis=1)
        flow_data.to_csv(f'{args.program_path}/{self.program_name}/AccuracyFlow.csv')
        revamp_relation=list(flow_data[flow_data['MRR_flow']>0.0001].index)
        new_clause=final_program.clause[final_program.clause.map(lambda one:True if one.head.predicate in revamp_relation else False)].reset_index(drop=True)
        final_program=Program(self.program_name,new_clause)
        
        #根据给定的优化的逻辑程序，对原先的逻辑程序进行更新
        final_program_prolog=final_program.to_prolog('',is_contain_fact=False,is_difference_head=True,is_save=False)
        final_program_clingo=final_program.to_clingo('',is_contain_fact=False,is_save=False)
        old_program_prolog=pandas.read_csv(f'{self.path}/KnowledgeBase.pl',sep='|',names=['prolog'],header=None,encoding=args.encode)['prolog']
        old_program_clingo=pandas.read_csv(f'{self.path}/Program.lp',sep='|',names=['clingo'],header=None,encoding=args.encode)['clingo']
        for head_predicate in revamp_relation:
            old_program_clingo=old_program_clingo[~old_program_clingo.str.contains(f'{head_predicate}.*:-.*')]
            relation_program_clingo=final_program_clingo[final_program_clingo.str.contains(f'{head_predicate}.*:-.*')]
            old_program_clingo=pandas.concat((old_program_clingo,relation_program_clingo),axis=0).reset_index(drop=True)
            old_program_prolog=old_program_prolog[~old_program_prolog.str.contains(f'{head_predicate}.*:-.*')]
            relation_program_prolog=final_program_prolog[final_program_prolog.str.contains(f'{head_predicate}.*:-.*')]
            old_program_prolog=pandas.concat((old_program_prolog,relation_program_prolog),axis=0).reset_index(drop=True)
        old_program_prolog.to_csv(f'{self.path}/KnowledgeBase.pl',sep='|',index=False,header=False,encoding=args.encode)
        old_program_clingo.to_csv(f'{self.path}/Program.lp',sep='|',index=False,header=False,encoding=args.encode)

if __name__=='__main__':
    pass

    # enhance_data=interacte.enhance_data()
    # print(enhance_data)
    # sys.exit()

    time1=time.time()

    app=Ragger(f'{args.dataset_name}_prog1',new_model=True,new_enhance=True)
    app.main(continue_mode=False)
    app.test()
    # app.revamp('Hit@1',0.65)

    time2=time.time()
    print(time2-time1)

    # p=profile.Profile()
    # p.run("app.test()")

    # x=pstats.Stats(p)
    # x.sort_stats('cumtime').print_stats(0.1)
