# 数据存放与接入

正式研究数据集尚未获取；polyhaven_demo子目录包含3张用于工程演示的真实HDR，不能代替正式训练集。目录按用途划分，数据集名称仅作为子目录名；下文 indoor_hdr 是示例名。

```text
data/
  raw/<dataset>/                 原始下载包、解压文件、授权和原始元数据
  manifests/<dataset>/           原始全景清单 sources.csv、来源和标定说明
  processed/<dataset>/<version>/ 预处理 manifest.jsonl、输入、HDR目标、预览和划分记录
  tokens/<dataset>/<version>/    未来的token缓存，绑定tokenizer权重与数据版本
runs/<run_id>/                   训练配置、日志、checkpoint与评估输出
```

原始数据保持不变；目录初始化和索引工具不会改写原始文件。若磁盘不足，可将原始数据存放在外置目录，并在 sources.csv 中填写绝对路径。不要将实际数据、缓存或权重提交到Git。

## 选择数据的必要条件

第一阶段需要室内、线性浮点、完整360度的HDR环境图，以及场景/房间分组信息。优先使用有清晰许可、官方划分、曝光与辐射标定说明的数据。普通LDR全景或RGB-D图片不能直接当作HDR光照真值。目录和扩展名不能证明辐射标定有效。

目前的适配器仅接受 H×2H×3 的ERP环境图，格式为 .hdr、.exr 或RGB float .npy。其他投影、通道或颜色空间需要额外适配；EXR解码支持需在实际文件上确认。

选择后在 manifests/<dataset>/ 中记录：下载地址、版本、获取日期、许可、RGB颜色空间、线性/非线性、数值单位、曝光/标定信息、房间分组依据和官方划分。未知项明确写“未核实”。

## 接入步骤

在项目根目录使用项目虚拟环境运行：

```powershell
.venv/Scripts/python.exe scripts/data_workspace.py init --dataset indoor_hdr
# 将数据放入 data/raw/indoor_hdr/ 后：
.venv/Scripts/python.exe scripts/data_workspace.py index --dataset indoor_hdr
```

索引会生成 data/manifests/indoor_hdr/sources.csv，且拒绝覆盖已有清单。它只发现候选文件，不判断图像内容是否有效。请填写每一行的 scene_id；同一房间的所有全景使用相同ID。若存在官方划分，填写 split=train/val/test；否则所有 split 留空，由准备程序按场景划分。至少需要三个独立场景组。

CSV的 pano_path 相对于CSV所在目录解析，也接受绝对路径。请勿用全景文件名直接替代房间ID，以免同一房间泄漏到不同集合。

```powershell
.venv/Scripts/python.exe scripts/data_workspace.py check --sources data/manifests/indoor_hdr/sources.csv
.venv/Scripts/python.exe scripts/prepare_data.py --sources data/manifests/indoor_hdr/sources.csv --out-dir data/processed/indoor_hdr/v1
```

check 只检查路径、重复记录与分组；prepare 才解码并检查非负有限HDR、重复图像和投影，生成配对数据及来源审计。prepare 拒绝写入非空输出目录。先核对预览的朝向、视场、曝光及亮源，再开始训练。默认 hdr_scale=1.0 只是数值变换配置，不代表数据已经物理标定。

## 数据准备后的训练

```powershell
.venv/Scripts/python.exe experiment.py --stage tokenizer --manifest data/processed/indoor_hdr/v1/manifest.jsonl --out_dir runs/tokenizer_v1
.venv/Scripts/python.exe experiment.py --stage ar --manifest data/processed/indoor_hdr/v1/manifest.jsonl --tokenizer runs/tokenizer_v1/best.pt --out_dir runs/ar_v1
.venv/Scripts/python.exe experiment.py --stage evaluate --manifest data/processed/indoor_hdr/v1/manifest.jsonl --checkpoint runs/ar_v1/best.pt --tokenizer runs/tokenizer_v1/best.pt --out_dir runs/ar_v1_test
```

训练需要另外安装匹配的PyTorch/TorchVision，并验证实际GPU计算。图像编码器预训练权重首次训练时可能需要下载。token缓存目录目前预留，现有AR训练会现场编码目标，尚未实现缓存。

当前固定视场、零俯仰和相机光心输出属于首阶段范围；真实随拍泛化、任意三维位置光照和弱监督尚未完成。
