#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/../../'))
if rpath not in sys.path:sys.path.append(rpath)

import clingo,pandas,tqdm,time,copy

from ragger.config import args
from ragger.model.ckyrule.core import Literal, Rule

class Generator(object):

    def __init__(self,head_predicate,
                      body_predicate,
                      self_link_relation,
                      undirected_relation,
                      is_diff_varcons=False):
        
        self.head_predicate=head_predicate
        self.body_predicate=body_predicate
        self.self_link_relation=set(self_link_relation)
        self.undirected_relation=set(undirected_relation)
        self.is_diff_varcons=is_diff_varcons

        self.var_lis=[f'X{i}' for i in range(1,args.max_specialize_body+2)]
        self.rulePattern=[(n,i) for n in range(1,args.max_body+1) for i in range(1,n+2)]

        #记录当前生成的所有约束，以便加入下一轮迭代中
        self._cons=''
        #记录当前主迭代过程的规则条件与变量数量
        self._body,self._var=0,0

    def _getClingoControl(self,body,var,is_specialize=False):

        #暂时无法将规则模板与实例融合生成，因为规则模板的答案集需要排序，从小到大依次生成，还没在clingo中找到解决途径

        arguments=[f'-c relationNum={len(self.body_predicate)}',]
        control=clingo.Control(arguments)
        control.configuration.solve.models = 0
        if is_specialize:control.load(f'{rpath}/{rname}/model/ckyrule/rlptn_splz.lp')
        else:control.load(f'{rpath}/{rname}/model/ckyrule/rlptn.lp')
        #如果头部谓词存在selfLink，则加入selfLink(-1).
        if self.head_predicate in self.self_link_relation:control.add("base", [], 'selfLink(-1).')
        #加入selfLink，监测存在自链接的关系，用以形成约束，减少搜索范围
        for relation in self.self_link_relation:
            if relation in self.body_predicate:control.add("base", [], f'selfLink({self.body_predicate.index(relation)}).')
        #加入undirected，监测无向关系，用以形成约束，减少搜索范围
        for relation in self.undirected_relation:
            if relation in self.body_predicate:control.add("base", [], f'undirected({self.body_predicate.index(relation)}).')

        #加入headRelation谓词，表示规则结论所代表的关系索引，为了避免规则条件中的谓词与规则结论中完全相同
        control.add("base", [], f'headRelation({self.body_predicate.index(self.head_predicate)}).')
        #加入一轮rulePattern谓词开启一轮搜索
        control.add("base", [], f'rulePattern({body},{var}).')
        #加入约束
        control.add("base", [], self._cons)

        return control

    def _getRuleExpress(self,atoms,var_num):

        body=[]
        for one_pattern in atoms:
            #为-1表示这个one_pattern是规则结论表示
            if one_pattern.arguments[2]==-1:head=Literal(self.head_predicate,(self.var_lis[one_pattern.arguments[1][0]],self.var_lis[one_pattern.arguments[1][1]]))
            #为0与1表示这个one_pattern是规则条件表示
            else:body.append(Literal(self.body_predicate[one_pattern.arguments[3]],(self.var_lis[one_pattern.arguments[1][0]],self.var_lis[one_pattern.arguments[1][1]])))

        #添加变量的约束，默认每个变量互不相等
        for n in range(0,var_num):
            for i in range(n+1,var_num):
                body.append(Literal('unequal',(self.var_lis[n],self.var_lis[i])))

        return Rule(head,body)

    def _search(self,control,var_num):

        control.ground([("base", [])])
        with control.solve(yield_=True) as handle:
            while True:
                handle.resume()
                model=handle.model()
                if model!=None:
                    atoms=model.symbols(shown = True)
                    atoms=sorted(atoms,key=lambda one:one.arguments[0].number)
                    atoms=pandas.Series(atoms).map(lambda one:Literal(one.name,(one.arguments[0].number,(one.arguments[1].arguments[0].number,one.arguments[1].arguments[1].number),one.arguments[2].number,one.arguments[3].number)))
                    # print([one.__str__() for one in atoms])
                    yield self._getRuleExpress(atoms,var_num)
                else:break
    
    def __iter__(self):

        #从前到后各有0,18,0,612,3564,0,10405,356454,646380个
        for self._body,self._var in self.rulePattern:
            control=self._getClingoControl(self._body,self._var,is_specialize=False)
            rule_num=0
            for rule in self._search(control,self._var):
                rule_num+=1
                yield rule
            print('搜索了长度为%s,变量%s个的规则共%s个'%(self._body,self._var,rule_num))
            # input()

    def specializeRule(self,rule):

        if rule.rule_length>=args.max_specialize_body:return
        body_str,_,_=self._bodyToStr(rule,no_head=False,is_var=False)
        sbody=rule.rule_length+1
        for svar in range(rule.var_num,rule.var_num+2):
            control=self._getClingoControl(sbody,svar,is_specialize=True)
            for one in body_str:
                control.add("base", [], f'{one}.')
            rule_num=0
            for rule in self._search(control,svar):
                rule_num+=1
                yield rule
            print('特化了长度为%s,变量%s个的规则共%s个'%(sbody,svar,rule_num))
    
    def _bodyToStr(self,rule,no_head=False,is_var=False):

        body_str,body_args,var_cons=[],[],[]
        #将规则转换为ASP内部表示
        for i,atom in enumerate(rule.body):
            if atom.predicate=='unequal':continue
            body_args+=list(atom.arguments)
            var_cons+=[f'Var{i*2}',f'Var{i*2+1}']
            if is_var:body_str.append(f'relationArg(Index{i},(Var{i*2},Var{i*2+1}),_,{self.body_predicate.index(atom.predicate)})')
            else:
                if i<=(rule.var_num-2):body_str.append(f'relationArg({i},({self.var_lis.index(atom.arguments[0])},{self.var_lis.index(atom.arguments[1])}),0,{self.body_predicate.index(atom.predicate)})')
                else:                  body_str.append(f'relationArg({i},({self.var_lis.index(atom.arguments[0])},{self.var_lis.index(atom.arguments[1])}),1,{self.body_predicate.index(atom.predicate)})')

        if no_head==False:
            var_cons+=[f'Var{rule.rule_length*2}',f'Var{rule.rule_length*2+1}']
            body_args+=list(rule.head.arguments)
            if is_var:body_str.append(f'relationArg(-1,(Var{rule.rule_length*2},Var{rule.rule_length*2+1}),-1,-1)')
            else:     body_str.append(f'relationArg(-1,({self.var_lis.index(rule.head.arguments[0])},{self.var_lis.index(rule.head.arguments[1])}),-1,-1)')
        return body_str,body_args,var_cons

    def _bodyToConsStr(self,rule,no_head=False):

        body_str,body_args,var_cons=self._bodyToStr(rule,no_head=no_head,is_var=True)
        #将相等的变量化为集合放入same_lis中，这样集合内是相等约束，集合间是不等约束
        same_lis,diff_var=[],[]
        for left,right in zip(body_args,var_cons):
            try:
                var_index=diff_var.index(left)
                same_lis[var_index].append(right)
            except ValueError:
                diff_var.append(left)
                same_lis.append([right])
        #根据已经形成的same_lis，向body_str中加入对应的相等约束与不等约束
        for i,one in enumerate(same_lis):
            left=one[0]
            #形成相等约束
            for right in one[1:]:
                body_str.append(f'{left}={right}')
            #形成不等约束
            for right in same_lis[i+1:]:
                body_str.append(f'{left}!={right[0]}')
        
        return body_str
    
    def _toCons(self,body_str):

        body_str=', '.join(body_str)
        cons=f':- {body_str}.\n'
        self._cons+=cons

    def ruleToCons(self,rule,no_head=False):

        if rule.rule_length>args.max_body:return
        if rule.rule_length==args.max_body and no_head==False:return

        body_str=self._bodyToConsStr(rule,no_head)
        self._toCons(body_str)

    def varToCons(self,rule,var_name,con_type):

        if rule.rule_length>=args.max_body:return

        var_index_lis=[]
        for i,atom in enumerate(rule.body):
            if atom.predicate=='unequal':continue
            left,right=atom.arguments
            if left==var_name:var_index_lis.append(f'Var{i*2}')
            if right==var_name:var_index_lis.append(f'Var{i*2+1}')

        def func(body_str,left,symbol,right):

            body_str_c=copy.deepcopy(body_str)
            body_str_c.append(f'Var{left}{symbol}{right}')
            #将新约束加入下一轮ASP内置搜索过程中
            self._toCons(body_str_c)

        body_str=self._bodyToConsStr(rule,True)
        body_str.append(f'relationArg(-1,(Var{rule.rule_length*2},Var{rule.rule_length*2+1}),-1,-1)')
        for var_index in var_index_lis:
            if self.is_diff_varcons:
                if 'head' in con_type:func(body_str,rule.rule_length*2,'!=',var_index)
                if 'tail' in con_type:func(body_str,rule.rule_length*2+1,'!=',var_index)
            else:
                if 'head' in con_type:func(body_str,rule.rule_length*2,'=',var_index)
                if 'tail' in con_type:func(body_str,rule.rule_length*2+1,'=',var_index)

def testRule(rule_str,show_str,var_num):

    rule_str=rule_str[:-1]
    var_lis=[f'X{i}' for i in range(1,var_num+1)]
    for n in range(len(var_lis)-1):
        for i in range(n+1,var_num):
            rule_str+=f', {var_lis[n]}!={var_lis[i]}'
    rule_str=rule_str+'.'

    control=clingo.Control()
    control.load(f'{rpath}/{rname}/model/ckyrule/input/bk.lp')
    control.add("base", [], rule_str)
    control.add("base", [], show_str)
    control.ground([("base", [])])
    with control.solve(yield_=True) as handle:
        while True:
            handle.resume()
            model=handle.model()
            if model!=None:
                atoms=model.symbols(shown = True)
                print(len(atoms))
            else:break


if __name__=='__main__':
    pass

    # gen=Generator('also_see',['hyponym','hypernym','member_holonym','also_see',
    #                           'instance_hypernym','member_meronym','member_of_domain_topic'])
    # for i in gen:
    #     print(i)

    testRule('also_see(X4,X1,1):- hyponym(X1,X2), hypernym(X3,X1), hypernym(X4,X1).',
             '#show also_see/3.',
             4)

    # time1=time.time()
    # for i in range(100):
    #     testRule('also_see(X2,X1,1):- also_see(X1,X2), also_see(X3,X1), member_holonym(X4,X3).',
    #             '#show also_see/3.',
    #             4)
    # time2=time.time()
    # print(time2-time1)