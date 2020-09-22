# NAS-deliver2

### 搜索启动
1. `mkdir data` 创建 data文件夹
2. 将目标数据集移动到data目录下，或`ln -s` 建立软连接
3. 调整并确认nas_config.json中的参数符合自己的搜索需求
4. `python nas.py`启动搜索

### 配置文件参数说明（nas_config.json）
配置文件修改较频繁的参数含义如下：   
```
nas_main：   
    num_gpu：可用gpu个数   
    block_num：需要搜索block个数    
    num_opt_best：train_winner阶段的采样次数   
    spl_network_round：每轮每个网络的采样次数   
    max_player_nums：经过初筛后参与竞赛的网络最大个数   
    add_data_per_round：每轮添加数据量   
    add_data_for_confirm_train：每个block的confim阶段需要数据量，调试时置100，搜索时置-1    
    repeat_num：每个block搜得结果的堆叠次数        
    pred_mask：不适用预测模块    
    filter_mask：不适用拓扑结构初筛模块   
    link_node：每个block末尾是否添加pooling   
enum：
    depth：每个block主链深度   
    width：支链个数   
    max_depth：支链最大深度   
eva：
    task_name：搜索的实际场景    
    search_epoch：搜索的epoch设定   
    confirm_epoch：confim阶段的epoch设定   
opt：
    sample_size：负例样本个数    
    positive_num：正例样本个数   
    uncertain_bit：每次采样可探索的编码位数   
spl:
    skip_max_dist：跨层连接最大距离   
    skip_max_num：跨层连接最大个数   
    space：节点操作的搜索空间   
```
### 已包含应用场景
- 分类（evaluator_classificatoin.py）
1. cifar10
2. cifar100
3. tiny-imagenet
- 去噪
1. waterloo,(evaluator_denoise.py)
2. SIDD,(evaluator_SIDD.py)

### 添加新的应用场景

1. 定义搜索场景。主搜索框架与评估的交互界面如下：主控传给评估task_item(EvaScheduleItem实例化的对象，一个实例对应一个任务，内部包含所有该任务的信息)，评估返回给主控评分（将评分填在task_item中返回），因此如需添加新的应用场景，将其定义为新的评估模块即可
2. 按照模板evaluator_user.py添加适合于自己特定应用场景的内容, 重命名为evaluator_***.py的格式
3. 在nas.py加入导入新应用场景评估文件的逻辑，将nas_config.json中task_name设置为新应用场景
4. 调整搜索配置参数及搜索空间
5. `python nas.py` 启动搜索

#### 经验加速模块的使用
- 经验加速模块分两部分：1、拓扑结构和节点操作的关系作为先验，2、拓扑结构之间的质量比较作为先验。使用经验加速模块，可以缩短搜索时长。是否使用经验加速模块，可通过配置文件中的pred_mask和filter_mask两个变量来设置
- 经验模块的代码针对所有应用场景是通用的，但经验模型不是通用的，如添加新的应用场景需要对经验模块的模型进行更新训练（经验模块已经适配代码中已包含的应用场景）。现对经验模块的训练接口和训练数据编码格式做如下说明，以便用户更新先验模型。



