#const relationNum=3.

% selfLink(1).
% undirected(2).
% headRelation(1).
% rulePattern(3,3).

% relationArg(0,(0,1),0,1).
% relationArg(1,(2,0),1,1).
% relationArg(-1,(2,1),-1,-1).

patternFrame(Index,(Value1,Value2)):- relationArg(Index,(Value1,Value2),0,Value3).
patternGraph(Index,(Value1,Value2)):- relationArg(Index,(Value1,Value2),1,Value3).

{patternFrame(Body-1,(Value1,Value2)): Value1=0..Var-2, Value2=Var-1; 
 patternFrame(Body-1,(Value1,Value2)): Value1=Var-1, Value2=0..Var-2} = 1:- rulePattern(Body,Var), {relationArg(Index,(_,_),0,_): Index<Body-1}=Frame, Var>Frame+1.
{patternGraph(Body-1,(Value1,Value2)): Value1=0..Var-1, Value2=0..Var-1} = 1:- rulePattern(Body,Var), {relationArg(Index,(_,_),0,_): Index<Body-1}=Frame, Var=Frame+1.

patternArg(Index,(Value1,Value2),0):- patternFrame(Index,(Value1,Value2)).
patternArg(Index,(Value1,Value2),1):- patternGraph(Index,(Value1,Value2)).

{relationArg(Index,(Value1,Value2),Type,Value3): Value3=0..relationNum-1} = 1:- patternArg(Index,(Value1,Value2),Type).

%在值允许的范围内，交换顺序得到的结构是一样的，无非叶节点的字母不同
:-patternFrame(Index,(Value1,Value2)), patternFrame(Index+1,(Value3,Value4)), Value1<Value2, Value3<Value4, Value1>Value3.
:-patternFrame(Index,(Value1,Value2)), patternFrame(Index+1,(Value3,Value4)), Value1>Value2, Value3>Value4, Value1>Value3.
% %在插入第一个谓词时，关系的方向是没有区别的
:-patternFrame(Index,(Value1,Value2)), Index=0, Value1>Value2.

%由于规则条件部分的图结构部分存在隆余，禁止关系的交换
:-relationArg(Index,(_,_),1,Value1), relationArg(Index+1,(_,_),1,Value2), Value1>Value2.
% %完全相同的原子公式模板，禁止填入相同的谓词
:-relationArg(Index1,(Value1,Value2),_,Value5), relationArg(Index2,(Value3,Value4),_,Value6), Index1!=Index2, Value1=Value3, Value2=Value4, Value5=Value6.
% %如果规则内部的谓词是自连接的，则去除
:-relationArg(-1,(Value1,Value2),-1,-1), headRelation(Value3), Value1=Value2, not selfLink(Value3).
:-relationArg(Index,(Value1,Value2),1,Value3), Value1=Value2, not selfLink(Value3).
% %如果规则内部的谓词是无向的，则只保留一个
:-relationArg(Index,(Value1,Value2),_,Value3), Value1>Value2, undirected(Value3).
% %如果规则条件中存在谓词与规则结论完全一致，则去除
:-relationArg(-1,(Value1,Value2),-1,-1), headRelation(Value3), relationArg(_,(Value4,Value5),_,Value6), Value1=Value4, Value2=Value5, Value3=Value6.

% #show rulePattern/2.
% #show patternFrame/2.
% #show patternGraph/2.
% #show patternArg/3.
#show relationArg/4.