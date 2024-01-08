#coding:utf-8
import sys,os
fpath,fname=os.path.split(__file__)
rpath,rname=os.path.split(os.path.abspath(fpath+'/'))
if rpath not in sys.path:sys.path.append(rpath)

import argparse

#基本设置
parser = argparse.ArgumentParser(description='项目所有参数定义')
parser.add_argument('--root_path',default=rpath,help = '项目根目录')
parser.add_argument('--encode',default='utf-8',help = '默认编码名称')
#数据集设置
parser.add_argument('--dataset_path',default=fpath+'/data',help = '数据集存储路径')
parser.add_argument('--dataset_name',default='WN18',help = '主体程序使用的数据名称')
parser.add_argument('--dataset_name_lis',default=['FB15k','FB15k-237','WN18','WN18RR'],help = '目前可用的数据集文件夹名称列表')
parser.add_argument('--program_path',default=fpath+'/program',help = '结果逻辑程序存储路径存储路径')
#程序运行选项设置
parser.add_argument('--is_enhance_dataset',default=True,help = '是否进行数据增强步骤，为False则使用原始数据')
parser.add_argument('--is_debug',default=False,help = '是否进入debug模式，目前就是多打印一些东西')
parser.add_argument('--is_sequential_covering',default=True,help = '是否使用序贯覆盖，即规则覆盖一部分数据，就删去那一部分')
parser.add_argument('--is_specialize_good',default=False,help = '得到评分为好的规则，是否继续进行specialize')
parser.add_argument('--is_diff_varcons',default=False,help = '是否使用不等变量约束，为False则使用相等变量约束，理论上相等变量约束是对的，但是慢很多')
#数据预增强模型设置
parser.add_argument('--model_path',default=fpath+'/model',help = '主体程序使用的模型文件路径')
parser.add_argument('--embed_model',default='TransE',help = """嵌入预增强使用的嵌入模型，目前可选[TransE,TransH,TransA,
                                                                                                TransD,KG2E,
                                                                                                SimplE,InteractE]""")
parser.add_argument('--candidate_entity_limit',default=3,help = '一个三元组嵌入预测排名的选取个数限制，取最好的几个出来进行后续处理')
parser.add_argument('--enhance_data_scale',default=2,help = '在原有数据集规模的基础上增强的比例')
#规则挖掘模型设置
parser.add_argument('--max_body',default=3,help = '默认ILP学习设置中搜索限制的最大规则条件中的谓词数量')
parser.add_argument('--max_specialize_body',default=3,help = '默认ILP学习设置中搜索限制的最大特化规则的谓词数量,需要不小于max_body')
#规则评估设置
parser.add_argument('--min_overlap_limit',default=0.03,help = '评估规则变量的最小重合度限制，变量高于此值通过评估')
parser.add_argument('--min_coverage_limit',default=0.03,help = '评估规则的最小覆盖率限制，规则高于此值通过评估')
parser.add_argument('--confidence_limit',default=0.03,help = '评估规则的正例置信度限制，规则高于此值通过评估，低于则需要特殊化')
parser.add_argument('--min_rule_score_limit',default=0.01,help = '评估规则的最小覆盖率限制，规则高于此值通过评估')
#规则排序评分设置
parser.add_argument('--confidence_inclined_coefficient',default=10,help = '计算规则评分时，着重考虑置信度的程度系数')
parser.add_argument('--gnl_inclined_coefficient',default=6,help = '计算规则评分时，着重考虑泛化指标的程度系数')

args=parser.parse_args()