#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/../../'))
if rpath not in sys.path:sys.path.append(rpath)

import pandas,operator,csv
from ragger.config import args

class Literal(object):

    def __init__(self, predicate, arguments, positive = True):

        assert isinstance(predicate,str),'predicate类型输入错误'
        assert isinstance(arguments,list) or isinstance(arguments,tuple),'arguments类型输入错误'
        assert isinstance(positive,bool),'positive类型输入错误'

        self.predicate = predicate
        self.arguments = arguments
        self.positive = positive
        self.arg_num = len(set(arguments))

    def __str__(self):

        x = self.predicate+str(self.arguments).replace(" ","").replace("'","")
        if not self.positive:
            x = 'not ' + x
        return x

    def __eq__(self, other):

        if self.predicate=='unequal' and other.predicate=='unequal' and set(self.arguments)==set(other.arguments):return True
        return self.__str__()==other.__str__()

    def __hash__(self):return hash(self.__str__())

class Rule(object):
    
    def __init__(self,head=None,body=[]):

        assert isinstance(head,Literal) or head==None,'head类型输入错误'
        assert isinstance(body,list) or isinstance(body,tuple),'body类型输入错误'
        for one in body:
            assert isinstance(one,Literal),'%s不是Literal类的对象'%one
        assert head!=None or body!=[],'head与body不能同时为空'

        self.head=head
        self.body=body
        self.var_num=len(set([arg for literal in body for arg in literal.arguments]))
        self.var_cons=[literal for literal in body if literal.predicate=='unequal']
        self.rule_length=len(self.body)-len(self.var_cons)

        if head==None:self.is_constrain=True
        else:self.is_constrain=False
        if self.rule_length==0:self.is_fact=True
        else:self.is_fact=False
        if head!=None and body!=[]:self.is_rule=True
        else:self.is_rule=False

    def __str__(self):

        head_str = ''
        if self.is_constrain==False:head_str = self.head.__str__()
        if self.is_fact==False:head_str+=':- '
        body_str = self.format_body()
        return f'{head_str}{body_str}.'

    def __eq__(self,other):

        if self.head!=other.head:return False
        if self.rule_length!=other.rule_length:return False
        for s_lit in self.body:
            exit_equal=False
            for o_lit in other.body:
                if s_lit==o_lit:exit_equal=True
            if exit_equal==False:return False
        return True

    def __hash__(self):return hash(self.__str__())

    def format_body(self):return ', '.join(literal.__str__() for literal in self.body if literal.predicate!='unequal')
    
    def format_rule(self):

        if self.var_cons==[]:return self.__str__()
        else:var_cons_str=', '.join(literal.__str__() for literal in self.var_cons)
        return f'{self.__str__()[:-1]}, {var_cons_str}.'

    def add_body(self,literal_lis):

        assert isinstance(literal_lis,Literal) or isinstance(literal_lis,list),f'type(literal_lis)={type(literal_lis)}错误，必须为Rule或者list'
        if isinstance(literal_lis,Literal):literal_lis=[literal_lis]
        #由于self.body中普通条件与不等条件是混在一起的，而在tester.groundRule中需要把不等条件放在最后，并且也是为了美观
        #所以这里需要把输入的literal_lis与self.body进行拆分
        body_literal=[one for one in self.body if one.predicate!='unequal']
        body_var_cons=self.var_cons
        new_literal=[one for one in literal_lis if one.predicate!='unequal']
        new_var_cons=[one for one in literal_lis if one.predicate=='unequal']
        return Rule(self.head,body_literal+new_literal+body_var_cons+new_var_cons)

class Program(object):
    
    def __init__(self,name=None,clause=[]):

        assert isinstance(name,str) or name==None,'name类型输入错误'
        assert isinstance(clause,list) or isinstance(clause,tuple) or isinstance(clause,pandas.Series),'clause类型输入错误'
        for one in clause:
            assert isinstance(one,Rule),'%s不是Rule类的对象'%one
            
        self.name=name
        self.clause=pandas.Series(clause).drop_duplicates()
        self.fact=self.clause[self.clause.map(operator.attrgetter('is_fact'))].reset_index(drop=True)
        self.rule=self.clause[self.clause.map(operator.attrgetter('is_rule'))].reset_index(drop=True)
        self.constrain=self.clause[self.clause.map(operator.attrgetter('is_constrain'))].reset_index(drop=True)
        self.clause=pandas.concat((self.fact,self.rule,self.constrain),ignore_index=True)

        self.fact_num=len(self.fact)
        self.rule_num=len(self.rule)
        self.constrain_num=len(self.constrain)
        self.clause_num=len(self.clause)

    def __str__(self):

        if self.name==None:name_str=''
        else:name_str=self.name
        rule_str='\n'.join(rule.__str__() for rule in self.rule)
        constrain_str='\n'.join(constrain.__str__() for constrain in self.constrain)
        return f'Program:{name_str}\n{rule_str}\n{constrain_str}'

    def format_program(self,is_show_fact=False,is_show_rule=True,is_show_constrain=True):

        if is_show_fact:fact_str='\n'.join(fact.__str__() for fact in self.fact)
        else:fact_str=''
        if is_show_rule:rule_str='\n'.join(rule.__str__() for rule in self.rule)
        else:rule_str=''
        if is_show_constrain:constrain_str='\n'.join(constrain.__str__() for constrain in self.constrain)
        else:constrain_str=''
        return f'{fact_str}\n{rule_str}\n{constrain_str}'

    def to_prolog(self,file_path,is_contain_fact=True,is_difference_head=True,is_save=True):

        if is_save:assert file_path[-3:]=='.pl','prolog文件需要以.pl结尾'

        prog=pandas.Series([],dtype=object)
        for i in range(len(self.rule)):
            rule=self.rule.loc[i]
            var_cons_str='| '.join(literal.__str__() for literal in rule.var_cons).replace('unequal(','').replace(')','').replace(',','\\=').replace('|',',')
            if is_difference_head:rule.head.predicate+='_r'
            if var_cons_str=='':prog.loc[i]=f'{rule.__str__()[:-1]}.'
            else:prog.loc[i]=f'{rule.__str__()[:-1]}, {var_cons_str}.'
            if is_difference_head:rule.head.predicate=rule.head.predicate[:-2]

        if is_contain_fact:prog=pandas.concat((self.fact,prog))
        if is_save:prog.to_csv(file_path,index=False,header=False,quoting=csv.QUOTE_NONE,sep='|',escapechar='',encoding=args.encode)
        else:return prog

    def to_clingo(self,file_path,is_contain_fact=True,is_save=True):

        if is_save:assert file_path[-3:]=='.lp','clingo文件需要以.lp结尾'
        prog=pandas.Series([],dtype=object)
        for i in range(len(self.rule)):
            rule=self.rule.loc[i]
            var_cons_str='| '.join(literal.__str__() for literal in rule.var_cons).replace('unequal(','').replace(')','').replace(',','!=').replace('|',',')
            if var_cons_str=='':prog.loc[i]=f'{rule.__str__()[:-1]}.'
            else:prog.loc[i]=f'{rule.__str__()[:-1]}, {var_cons_str}.'

        if is_contain_fact:prog=pandas.concat((self.fact,prog,self.constrain),ignore_index=True)
        else:prog=pandas.concat((prog,self.constrain))
        if is_save:prog.to_csv(file_path,index=False,header=False,quoting=csv.QUOTE_NONE,sep='|',escapechar='',encoding=args.encode)
        else:return prog

    def add_clause(self,clause=[]):

        assert isinstance(clause,list) or isinstance(clause,tuple) or isinstance(clause,Rule) or isinstance(clause,pandas.Series),'clause类型输入错误'
        for one in clause:
            assert isinstance(one,Rule),'%s不是Rule类的对象'%one
        if isinstance(clause,Rule):clause=[clause]
        if isinstance(clause,pandas.Series)==False:clause=pandas.Series(clause).drop_duplicates()
        clause=pandas.concat((self.clause,clause)).drop_duplicates().iloc[self.clause.shape[0]:].reset_index(drop=True)
        self.clause=pandas.concat((self.clause,clause),ignore_index=True)
        self.clause_num+=clause.shape[0]
        for one in clause:
            if one.is_fact:
                self.fact.loc[self.fact.shape[0]]=one
                self.fact_num+=1
            elif one.is_rule:
                self.rule.loc[self.rule.shape[0]]=one
                self.rule_num+=1
            elif one.is_constrain:
                self.constrain.loc[self.constrain.shape[0]]=one
                self.constrain_num+=1
            else:raise ValueError('存在未知错误')

    @classmethod
    def read(cls,file_path):return pandas.read_csv(file_path,header=None,names=['clause'],sep='	',encoding=args.encode).astype('str')['clause']

if __name__=='__main__':
    pass

    head=Literal('also_see',('A','D'))
    body1=[Literal('hyponym',('C','D')),Literal('instance_hypernym',('C','B')),Literal('also_see',('C','B'))]
    body2=[Literal('hyponym',('C','D')),Literal('also_see',('C','B')),Literal('instance_hypernym',('C','B'))]
    x=Rule(body=body1)
    y=Rule(body=body1,head=head)
    z=Rule(head=head)
    d=Rule(body=[head],head=head)
    q=Rule(body=body2)
    prog=Program('hhh',[x,x,y,x,z])
    prog.add_clause([y,y,d])
    # print(Literal('unequal',('C','B')))
    # print(prog)
    print(x)
    print(y)
    print(z)
    print(q)
    print(q in [y,z])
    print(Literal('unequal',('C','B'))==Literal('unequal',('C','C')))
    # print(prog.clause)
    # print(prog.fact)
    # print(prog.rule)
    # print(prog.constrain)
    # print(prog.clause_num)
    # print(prog.fact_num)
    # print(prog.rule_num)
    # print(prog.constrain_num)
    # prog.to_clingo('./test.lp')