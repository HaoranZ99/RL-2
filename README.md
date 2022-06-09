# island-survive
## 安装
- 创建一个虚拟环境，python版本为3.8.3
- 运行`batchinstall.py`，安装所依赖的包
- 进入`./custom-env`下，键入`pip install -e .`
## 训练
- 运行`islandsurvive.py`
- 训练结果存放在`./checkpoints`下
## 回放
- 在`./checkpoints`下选取相应的训练后模型作为命令行参数。如`checkpoints/2022-04-06T11-34-17/island_net_20.chkpt`
- 将上面选择的训练后的模型作为命令行参数，运行`replay.py`。如`python replay.py checkpoints/2022-04-06T11-34-17/island_net_20.chkpt`
- replay的结果存放在`./replays`下