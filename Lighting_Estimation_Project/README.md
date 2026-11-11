# 单幅图像自回归光照估计

目标：从室内LDR透视照片，用图像条件自回归模型恢复相机位置的完整HDR环境光照，评估方向、辐射分布、高亮能量和球体照明效果。

已有初始实现包含HDR投影与配对数据准备、VQ tokenizer、图像条件因果Transformer、两阶段训练、续训、完整生成评估和单图推理。正式研究数据集尚未获取。已用3张Poly Haven真实HDR跑通CPU两阶段训练与推理演示；预测质量仍不足以用于真实光照估计，尚无正式基准结果。

## 数据放在哪里

原始数据放在项目的 `data/raw/<数据集名称>/`。数据存放约定、候选数据要求、清单填写方法与训练命令见 [数据接入说明](data/README.md)。

先初始化通用目录，无需选择数据集，也无需安装PyTorch：

```powershell
.venv/Scripts/python.exe scripts/data_workspace.py init
```

数据预处理仅需要NumPy与OpenCV，独立入口为 `scripts/prepare_data.py`。训练与推理入口为 `experiment.py`，初始配置为 `configs/initial.json`；配置值是待实验验证的起点。

## 环境与实现边界

使用独立虚拟环境。完整依赖见 requirements.txt，GPU版torch与torchvision需要按官方支持选择匹配构建。当前虚拟环境是否安装完成以实际导入和算子运行为准。

tokenizer使用球面面积加权的log-HDR重建、线性能量与漫反射积分误差，AR阶段固定tokenizer并优化下一token交叉熵。验证通过完整贪心生成评估HDR，不能将teacher forcing准确率等同于光照恢复质量。

现有训练尚未实现AMP、token缓存和连续回归对照；不能将当前框架描述为已完成研究验证。

## 原有启发式演示

`main.py` 和 `src/engine.py` 是旧亮区检测演示，与可训练的自回归系统分开。运行示例：`python main.py -i inputs/test_image.jpg`。其输出是HDR-like示意图，不能作为真实HDR估计精度的证据。

## 完整训练演示

运行方式和结果边界见 [运行演示](docs/运行演示.md)。演示入口： scripts/run_demo.py；关键逻辑检查： scripts/verify_pipeline.py。

## 优化状态

重建质量改善，纯AR预测仍未建立有效输入依赖，新增可选的可见区域约束。实测与限制见 [优化记录](docs/优化记录.md)。

## 自回归受控实验

可见优先、前序扰动和预训练编码器三组实验已完成。预训练模型最后一轮通过全部96张训练样本的完整生成检查，陌生测试场景误差尚未改善。复现入口和实测见 [实验记录](docs/自回归受控实验.md)。
