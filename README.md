简述
====

这是文章“**Optimize rule mining based on constraint learning in knowledge graph**”的源代码，它可以用来进行知识图谱补全的任务。其中的子模块ckyrule是一个简单的ILP系统，能够针对特定的问题从而学习规则。但是这些规则是离散的，并且无法进行谓词创造。

## 依赖

* `SWI-Prolog` version 8.4.3
* `Python` version 3.9.6
* `clingo` version 5.5.2
* `pyswip` version 0.2.11

其他依赖详见`requirements.txt`。

## 使用

在下载代码之后首先需要解压数据集，其次在`config.py`里配置所需参数，在`main.py`里设置需要的运行模式，最后运行`main.py`即可。其中`main.py`的一些运行模式参数如下：

`new_model: bool`->是否训练新的嵌入模型

`new_enhance: bool`->是否进行新的数据增强

`continue_mode: bool`->是否使用继续模式(在一些情况下可能会导致程序异常退出，继续模式能够完成还未挖掘过的规则)

此外，可以在`main.py`中设置程序最终生成的规则库名称，这将会以文件夹形式保存在`/program/`文件夹下。

## 复现论文中的结果

或许你会需要复现论文中对比实验的结果，在此给出论文中的参数配置:

### WN18

* `max_body` 设置为 3
* `max_specialize_body` 设置为 3
* `min_overlap_limit` 设置为 1e-3
* `min_coverage_limit` 设置为 1e-3
* `confidence_limit` 设置为 5e-3
* `min_rule_score_limit` 设置为 5e-3

### FB15K

* `max_body` 设置为 2
* `max_specialize_body` 设置为 2
* `min_overlap_limit` 设置为 1e-3
* `min_coverage_limit` 设置为 1e-3
* `confidence_limit` 设置为 0
* `min_rule_score_limit` 设置为 1e-3

### WN18RR

* `max_body` 设置为 4
* `max_specialize_body` 设置为 4
* `min_overlap_limit` 设置为 1e-7
* `min_coverage_limit` 设置为 1e-8
* `confidence_limit` 设置为 1e-3
* `min_rule_score_limit` 设置为 1e-3

### FB15K-237

* `max_body` 设置为 3
* `max_specialize_body` 设置为 3
* `min_overlap_limit` 设置为 0.5
* `min_coverage_limit` 设置为 1e-3
* `confidence_limit` 设置为 0
* `min_rule_score_limit` 设置为 1e-3

此外，对比实验中没有使用“rule base generation module”。因此，`is_enhance_dataset`设置为False。这些设置并不是四个数据集上最优的，只是我们探索过程中认为较好的结果。

并且代码中提供了函数`revamp(self,indicator,bad_rule_limit)`，能够根据设置的指标类型`indicator`，与期望的指标数值`bad_rule_limit`，重新运行代码。这能够根据不同的超参数设置从而适应不同类型的关系，例如小样本关系可以将超参数设置地宽泛一些。新得到的规则库若在MRR指标上存在些许提升，则这些规则都会更新进入最终的规则库中。

## 出版


## 引用Ragger


联络
====


许可证
======
