# NAS-deliver2

---------------------------

## 版本说明
### 搜索启动说明
1. mkdir data 创建 data文件夹
2. 将目标数据集移动到data目录下，或ln -s 建立软连接
3. 调整并确认nas_config.json中的参数符合自己的搜索需求
4. python nas.py启动搜索

### 配置文件参数说明（nas_config.json）

### 已包含应用场景
- 分类（evaluator_classificatoin.py）
1. cifar10
2. cifar100
3. tiny-imagenet
- 去噪
1. waterloo,(evaluator_denoise.py)
2. SIDD,(evaluator_SIDD.py)

### 添加新的应用场景
1. 按照模板evaluator_user.py添加适合于自己特定应用场景的内容, 重命名为evaluator_***.py的格式
2. 在nas.py加入导入新应用场景评估文件的逻辑，在nas_config.json设置在新应用场景下搜索
3. 如果不修改搜索空间，则跳过此步。如果需要修改搜索空间，可在nas_config.json中进行调整
4. 调整搜索配置参数
5. python nas.py 启动搜索

