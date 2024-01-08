#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/../../'))
if rpath not in sys.path:sys.path.append(rpath)

import time,pandas,copy
from ragger.config import args
from ragger.utils import DataManage
from ragger.model.ckyrule.core import Rule,Program
from ragger.model.ckyrule.tester import Tester
from ragger.model.ckyrule.generate import Generator

class CKYRule(object):

    def __init__(self,data_manage=None,
                      is_debug=False,
                      is_sequential_covering=True,
                      is_specialize_good=False,
                      is_diff_varcons=False):

        if type(data_manage)!=type(None):self.data_manage=DataManage(args.dataset_name)
        else:self.data_manage=data_manage

        #得到存在自连接的关系
        self.self_link_relation=self.data_manage.data_statistic['train_self_link_relation']
        #得到无向关系
        self.undirected_relation=[]
        undirected_rate=len(self.data_manage.data)/(len(self.data_manage.data)+(len(self.data_manage.valid_data)+len(self.data_manage.test_data))*3)
        for key,value in self.data_manage.data_statistic['train_relation_undirected_rate'].items():
            if value>undirected_rate:self.undirected_relation.append(key)

        #是否开启调试模式(现在主要是打印搜索过程)
        self.is_debug=is_debug
        #是否使用序贯覆盖
        self.is_sequential_covering=is_sequential_covering
        #是否特化高质量规则，解决一个实体链接多个实体情况下的推理排序问题
        self.is_specialize_good=is_specialize_good
        #是否使用不等变量约束，按理来说相等变量约束才是对的，但是意外发现不等变量约束比后者快太多
        self.is_diff_varcons=is_diff_varcons

        #记录正在特化的规则列表
        self._specialize_rule_lis=[]
        #保存需要specialize的一条规则，因为需要先过一遍一跳规则，生成变量约束，从而节省specialize时间
        self._prepare_specialize_lis=[]
        #保存一跳规则的实例化结果，空间换时间，可在多次迭代中使用
        self._relation_data_series=pandas.Series([])
        #保存多条规则的实例化结果，空间换时间，大部分只在本轮迭代中有效，但还是都存起来，加速refine_rule的速度
        self._ground_data_series=pandas.Series([])

    def evaluateRule(self,ground_head,rule):

        #规则实例化数量为零则直接返回并生成对应约束
        if len(ground_head)==0:return 'no_head_constrain',0

        #评估规则的正例覆盖率与置信度
        pos_coverage,pos_confidence,pos_accuracy_data=self.tester.evaluateRule(ground_head,'pos')
        if self.is_debug:print(f'pos coverage:{pos_coverage}')
        if self.is_debug:print(f'pos confidence:{pos_confidence}')

        if pos_coverage<=args.min_coverage_limit:process_type='constrain'
        elif pos_confidence>args.confidence_limit:process_type='good'
        else:
            if self.generator._body==1:process_type='prepare_specialize'
            else:process_type='specialize'
        if self.is_debug:print(process_type)

        return process_type,pos_accuracy_data

    #评估规则的各实例化变量列表是否与目标谓词集合有重叠的可能，无可能则直接'constrain'
    def evaluateVar(self,ground_var):
        
        head_overlap_rate=self.tester.evaluateVar(ground_var,'head')
        tail_overlap_rate=self.tester.evaluateVar(ground_var,'tail')

        if head_overlap_rate<=args.min_overlap_limit:head_process_type='constrain'
        else:head_process_type='None'
        if tail_overlap_rate<=args.min_overlap_limit:tail_process_type='constrain'
        else:tail_process_type='None'

        return head_process_type,tail_process_type

    def processRule(self,rule,good_rule_lis):

        if self.is_debug:print(rule)
        #实例化规则
        ground_var,ground_rule=self.tester.groundRule(rule)
        ground_head=self.tester.groundHead(rule,ground_rule)
        #规则评估
        process_type,_=self.evaluateRule(ground_head,rule)
        #对不同的process_type进行不同的处理
        if process_type=='prepare_specialize':self._prepare_specialize_lis.append(rule)
        elif process_type=='no_head_constrain':self.generator.ruleToCons(rule,no_head=True)
        elif process_type=='constrain':self.generator.ruleToCons(rule)
        elif process_type=='specialize':self._specialize_rule_lis.append(rule)
        elif process_type=='good':
            self._ground_data_series[rule.format_body()]=ground_rule
            good_rule_lis.append(rule)
            if self.is_specialize_good==False:self.generator.ruleToCons(rule)
            if self.is_debug==False:print('发现规则%s'%rule)
        else:raise ValueError('process_type=%s错误，无法取值'%process_type)

        #变量评估
        for name,one in ground_var:
            head_process_type,tail_process_type=self.evaluateVar(one)
            if head_process_type=='constrain' and tail_process_type=='constrain':
                self.generator.varToCons(rule,name,con_type='head_tail')
            elif head_process_type=='constrain':
                self.generator.varToCons(rule,name,con_type='head')
            elif tail_process_type=='constrain':
                self.generator.varToCons(rule,name,con_type='tail')
            else:pass

    def getRuleScore(self,pos_coverage,pos_confidence,gnl_coverage,gnl_confidence):

        sum_number=1+args.confidence_inclined_coefficient+args.gnl_inclined_coefficient*(1+args.confidence_inclined_coefficient)
        return (pos_coverage+args.confidence_inclined_coefficient*pos_confidence+ \
               args.gnl_inclined_coefficient*(gnl_coverage+args.confidence_inclined_coefficient*gnl_confidence))/sum_number
    
    def calculateRuleScore(self,rule):

        ground_rule=self._ground_data_series[rule.format_body()]
        ground_head=self.tester.groundHead(rule,ground_rule)
        pos_coverage,pos_confidence,pos_accuracy_data=self.tester.evaluateRule(ground_head,'pos')
        gnl_coverage,gnl_confidence,gnl_accuracy_data=self.tester.evaluateRule(ground_head,'gnl')
        if pos_accuracy_data.empty:rule_score=0.0
        else:rule_score=self.getRuleScore(pos_coverage,pos_confidence,gnl_coverage,gnl_confidence)
        return rule_score,pos_accuracy_data

    def refineRule(self,good_rule_lis,program_rule_lis):

        if good_rule_lis==[]:return
        rule_score_data=pandas.DataFrame([],columns=['rule','rule_score','pos_accuracy_data'])
        for rule in good_rule_lis:
            rule_score_data.loc[rule.__str__(),:]=[rule,None,None]
        rule_score_data['rule_score']=rule_score_data['rule_score'].astype(float)
        is_init=False
        while True:
            if is_init==False or self.is_sequential_covering:
                rule_score_data.iloc[:,:]=rule_score_data.apply(lambda line:(line['rule'],*self.calculateRuleScore(line['rule'])),axis=1,result_type='expand')
                is_init=True

            rule_score_data=rule_score_data[rule_score_data['rule_score']>=args.min_rule_score_limit]
            if rule_score_data.empty:break

            best_rule_index=rule_score_data['rule_score'].idxmax()
            best_rule,max_rule_score,best_rule_data=rule_score_data.loc[best_rule_index,:]
            if self.is_debug==False:print('输出评分为%s的规则%s'%(max_rule_score,best_rule))
            if  best_rule not in program_rule_lis:
                self.generator.ruleToCons(best_rule)
                program_rule_lis.append(best_rule)
                if self.is_sequential_covering==True:self.tester.pos_data=self.tester.drop_coverage_example(best_rule_data,self.tester.pos_data)
            rule_score_data=rule_score_data.drop(best_rule_index)
            if rule_score_data.empty:break

    def _run_func(self,rule,good_rule_lis):

        self.processRule(rule,good_rule_lis)
        
        rule_con_lis=[]
        while self._specialize_rule_lis!=[]:
            n=0
            rule=self._specialize_rule_lis.pop(0)
            rule_con_lis.append(rule)
            if self.is_debug==False and rule.rule_length<args.max_specialize_body:print('特化规则"%s"...'%rule)
            for srule in self.generator.specializeRule(rule):
                n+=1
                if self.is_debug==False and n%50==0:print('已特化%s条规则...'%n)
                self.processRule(srule,good_rule_lis)
            if self.is_debug==False and rule.rule_length<args.max_specialize_body:print('还有%s条规则待特化...'%len(self._specialize_rule_lis))
        if rule_con_lis!=[]:self.generator.ruleToCons(rule_con_lis[0])

    def run(self,app_name=None):

        good_rule_lis,program_rule_lis=[],[]
        n=0
        for rule in self.generator:
            n+=1
            if self.is_debug:print(rule)
            elif n%50==0:print('已搜索%s条规则...'%n)
            #截留规则长度为1的规则的特化，留到生成规则长度为2时再进行，这是为了更新规则长度为1的约束到搜索器中，节约时间
            if self.generator._body>1:
                while self._prepare_specialize_lis!=[]:
                    one=self._prepare_specialize_lis.pop(0)
                    self._run_func(one,good_rule_lis)
            self._run_func(rule,good_rule_lis)
        #如果最大规则长度为1，则需要在迭代过程外进行遗留的待特化规则的特化
        if args.max_body==1:
            while self._prepare_specialize_lis!=[]:
                self.generator._body=2
                one=self._prepare_specialize_lis.pop(0)
                self._run_func(one,good_rule_lis)

        self.refineRule(good_rule_lis,program_rule_lis)
        return Program(app_name,program_rule_lis)
    
    def selectBodyPredicate(self,bk_data,relation):

        entity_relation_set={relation}
        for _ in range(args.max_specialize_body):
            relation_data=bk_data[bk_data['relation'].map(lambda one:True if one in entity_relation_set else False)]
            relation_entity_set=set(pandas.concat((relation_data['head'],relation_data['tail'])))
            entity_relation_set=set(bk_data[bk_data['head'].map(lambda one:True if one in relation_entity_set else False) | \
                                            bk_data['tail'].map(lambda one:True if one in relation_entity_set else False) \
                                            ]['relation'])
        return list(entity_relation_set)

    def search(self,head_predicate):

        bk_data,pos_data,gnl_data=self.data_manage.generateData(head_predicate)
        body_predicate=self.selectBodyPredicate(bk_data,head_predicate)
        #由于序贯覆盖的原因，pos_data可能会改变
        self.tester = Tester(bk_data,copy.deepcopy(pos_data),gnl_data,self._relation_data_series)
        self.generator = Generator(head_predicate,body_predicate,self.self_link_relation,self.undirected_relation,is_diff_varcons=self.is_diff_varcons)

        #由于未知的错误，self._prepare_specialize_lis有可能会积累到下一轮搜索从而引发错误，在这里进行清除
        self._prepare_specialize_lis.clear()
        #由于_ground_data_series基本上只在refine_rule时起作用，在新一轮迭代时重置以释放内存
        self._ground_data_series=pandas.Series([])
        return self.run()

if __name__=='__main__':
    pass

    head_predicate='also_see'
    body_predicate=['derivationally_related_form', 'has_part', 'hypernym', 'hyponym', 
                    'instance_hypernym', 'instance_hyponym', 'member_holonym', 'member_meronym', 
                    'member_of_domain_region', 'member_of_domain_topic', 'member_of_domain_usage',
                    'part_of', 'similar_to', 'synset_domain_region_of', 'synset_domain_topic_of',
                    'synset_domain_usage_of', 'verb_group']

    # time1=time.time()
    # app=CKYRule(is_debug=True,)
    # program=app.run()
    # time2=time.time()
    # print(program)
    # print('time=%s'%(time2-time1))