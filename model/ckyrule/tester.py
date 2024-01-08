#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/../../'))
if rpath not in sys.path:sys.path.append(rpath)

import pandas,clingo,copy,numpy
from ragger.model.ckyrule.core import Literal,Rule

class Tester(object):

    def __init__(self,bk_data,pos_data,gnl_data,_relation_data_series):

        self.bk_data=bk_data
        self.pos_data=pos_data
        self.gnl_data=gnl_data
        self._relation_data_series=_relation_data_series

        #用于放弃计算过程种占用内存过大的规则
        self._max_ground_data_num=len(self.bk_data)*5
        #用于保存中间结果，evaluateRule函数评估'pos'与'gnl'需要连续
        self._accuracy_size=0

    def groundRule(self,rule):
        """
        :Doc->在bk的基础上，对规则进行实例化
        :Parameters(rule:tuple)->要实例化的规则
        :Return(pandas.DataFrame)->规则的实例化数据
        """
        # head=Literal('also_see',('A','D'))
        # body=[Literal('derivationally_related_form',('A','B')),
        #       Literal('derivationally_related_form',('B','C')),
        #       Literal('derivationally_related_form',('C','D'))]

        ground_data=[]
        exit_var=set([])
        for literal in rule.body:
            relation,arguments=literal.predicate,literal.arguments
            #不等于谓词一般放在最后面，所以在这个位置跳过
            if relation=='unequal':continue
            arguments_set=set(arguments)
            #加速，空间换时间，一跳规则的实例化是不变的
            try:rdata=copy.deepcopy(self._relation_data_series[relation])
            except KeyError:
                rdata=self.bk_data[self.bk_data['relation']==relation].drop(labels='relation',axis=1).reset_index(drop=True)
                self._relation_data_series[relation]=rdata
            rdata.columns=arguments
            #筛选内部约束
            if len(arguments_set)!=len(arguments):
                drop_data=pandas.Series([True]*rdata.shape[0])
                for var in set(rdata.columns):
                    dup_data=rdata[var]
                    #意味着存在重复变量
                    if isinstance(dup_data,pandas.DataFrame):
                        for i in range(dup_data.shape[1]-1):
                            drop_data&=(dup_data.iloc[:,i]==dup_data.iloc[:,i+1])
                rdata=rdata[drop_data].iloc[:,(~pandas.Series(arguments).duplicated()).tolist()]
            #筛选外部约束
            bind_var=arguments_set & exit_var
            if bind_var!=set([]):
                no_bind_var=arguments_set - exit_var
                exit_var.update(no_bind_var)
                bind_index=[]
                #链接新加入的数据
                while bind_var!=set([]):
                    for i,one_data in enumerate(ground_data):
                        join_var=set(one_data.columns) & bind_var
                        if join_var!=set([]):
                            try :
                                ground_data[i]=one_data.join(rdata.set_index(list(join_var)),on=list(join_var),how='inner').reset_index(drop=True)
                                bind_var-=join_var
                                bind_index.append(i)
                            except Exception:
                                #计算过程内存不够，跳过
                                return [],pandas.DataFrame([])
                #合并变动的数据
                for i in bind_index[1:]:
                    join_var=set(ground_data[bind_index[0]].columns) & set(ground_data[bind_index[i]].columns)
                    ground_data[bind_index[0]]=ground_data[bind_index[0]].join(ground_data[bind_index[i]].set_index(list(join_var)),on=list(join_var),how='inner').reset_index(drop=True)
                for i in reversed(bind_index[1:]):
                    ground_data.pop(i)
            #如果新数据没有相同变量，则加入ground_data
            else:
                exit_var.update(arguments_set)
                ground_data.append(rdata)
            #如果ground_data过大，则放弃此规则，因为内存不允许
            if len(ground_data[0])>=self._max_ground_data_num:return [],pandas.DataFrame([])

        #处理规则中的不等约束
        for literal in rule.var_cons:
            arguments=literal.arguments
            data=ground_data[0].loc[:,arguments]
            ground_data[0]=ground_data[0][data[arguments[0]]!=data[arguments[1]]]

        ground_rule=ground_data[0]
        ground_var=[[ground_rule[column].name,set(ground_rule[column].tolist())] for column in ground_rule]
        return ground_var,ground_rule
    
    def groundHead(self,rule,ground_rule):

        if ground_rule.empty:return []
        ground_head=ground_rule.loc[:,rule.head.arguments].drop_duplicates(ignore_index=True)
        ground_head.columns=['head','tail']
        return ground_head

    def evaluateRule(self,ground_head,data_type):

        if data_type=='pos':relation_data=self.pos_data
        elif data_type=='gnl':relation_data=self.gnl_data
        else:raise ValueError('data_type=%s输入错误'%data_type)
        
        if relation_data.empty:return 0,0,pandas.DataFrame([])
        accuracy_data=pandas.concat((ground_head,relation_data.loc[:,['head','tail']]),axis=0)
        accuracy_data=accuracy_data[accuracy_data.duplicated()]
        accuracy_size=len(accuracy_data)
        
        #保存用于后续计算gnl的gound_rule_size，需要减去已经正确的三元组数量
        if data_type=='pos':self._accuracy_size=accuracy_size
        if data_type=='gnl':gound_rule_size=len(ground_head)-self._accuracy_size
        else:gound_rule_size=len(ground_head)
        if gound_rule_size==0:return 0,0,pandas.DataFrame([])
        relaiton_size=len(relation_data)

        coverage=accuracy_size/relaiton_size
        confidence=accuracy_size/gound_rule_size
        return coverage,confidence,accuracy_data

    def evaluateVar(self,ground_var,var_type):

        relation_data=self.pos_data
        overlap_data=relation_data[var_type].map(lambda line:True if line in ground_var else False)
        overlap_rate=len(relation_data[overlap_data])/len(relation_data)
        return overlap_rate

    def drop_coverage_example(self,pos_accuracy_data,cover_data):

        data=pandas.concat((cover_data.loc[:,['head','tail']],pos_accuracy_data),ignore_index=True)
        data=data[~data.duplicated(keep='last')].loc[:cover_data.shape[0]-1,:].reset_index(drop=True)
        return data

if __name__=='__main__':
    pass

    tester=Tester()
    head=Literal('also_see',('X4','X1'))
    b1=Literal('hyponym',('X1','X2'))
    b2=Literal('hypernym',('X3','X1'))
    b3=Literal('hypernym',('X4','X1'))
    b4=Literal('unequal',('X1','X2'))
    b5=Literal('unequal',('X1','X3'))
    b6=Literal('unequal',('X1','X4'))
    b7=Literal('unequal',('X2','X3'))
    b8=Literal('unequal',('X2','X4'))
    b9=Literal('unequal',('X3','X4'))
    body=[b1,b2,b3,b4,b5,b6,b7,b8,b9,]
    rule=Rule(head,body)
    ground_rule=tester.groundRule(rule)
    print(ground_rule)
    