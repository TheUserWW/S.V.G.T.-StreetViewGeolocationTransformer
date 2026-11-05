# S.V.G.T. Mapillary 预训练指南

预训练素材唯一来源为 **Mapillary**。OSV-5M、GSV-Cities、其它街景来源及混合素材均不再接受。训练集、验证集和续训都会逐条校验来源与原始 Mapillary 图片 ID。

## Token 与 API

从 [Mapillary 开发者门户](https://www.mapillary.com/dashboard/developers) 创建应用并取得 Client Access Token。参考 [官方 API 认证说明](https://mapillary.github.io/mapillary-python-sdk/docs/mapillary.models/mapillary.models.client/)。程序使用官方 `graph.mapillary.com` 查询图片，将 Token 放在 Authorization 请求头，只向 Mapillary API 发送；图片下载保留作者、来源与许可记录。

不需要把 Token 写进源代码或命令参数。命令行可用 `--prompt-token` 隐藏输入，或读取本机环境变量 `MAPILLARY_ACCESS_TOKEN`。

```powershell
# 查询可选国家/地区及内置行政区，不需要 Token
python -m streetgeo download --list-regions

# 按国家下载
python -m streetgeo download --prompt-token --countries CHN,JPN --output data/mapillary_east_asia --max-images 10000

# 按大洲下载；大洲代码：africa,asia,europe,north-america,south-america,oceania,antarctica
python -m streetgeo download --prompt-token --continents europe --output data/mapillary_europe

# 美国州或任意国家中的自定义区域
python -m streetgeo download --prompt-token --regions US-CA,US-NY --output data/mapillary_states
python -m streetgeo download --prompt-token --countries JPN --bbox 139.5 35.5 139.9 35.9 --output data/mapillary_tokyo
python -m streetgeo download --prompt-token --geojson data/region.geojson --output data/mapillary_region
```

下载器筛选 Mapillary 透视街景，跳过未投影的全景图和短边不足 224 像素的素材。优先下载指定分辨率，必要时尝试更大版本。下载质量、地理过滤与数据来源检查分别执行；实际可下载张数取决于 Mapillary 覆盖情况。

SQLite 保存图片 ID、已完成文件、分页游标、失败及延后重试任务。大范围先使用粗网格排队，查询时再细分。429 和临时错误有有界重试，过大查询区域自动细分。API 持续异常时安全暂停，不会自动开始训练。`--max-pages` 只限制本次查询数；已下载数据保留。`--max-gb` 是停止阈值，并发批次可能略微超出。

## 数据划分与训练

```powershell
python -m streetgeo prepare --dataset-root data/mapillary_east_asia --output-dir data/world_mapillary
python -m streetgeo train --train-manifest data/world_mapillary/train.jsonl --val-manifest data/world_mapillary/val.jsonl --model-config configs/streetgeo_400m.json --output checkpoints/streetgeo_400m_mapillary/best.pt --epochs 4 --batch-size 32 --num-workers 4 --device cuda --amp bf16
```

也可一步完成下载、划分、训练：

```powershell
python -m streetgeo train --download --prompt-token --countries JPN --download-output data/mapillary_japan --data-output data/mapillary_japan_split --max-images 10000 --model-config configs/streetgeo_400m.json --output checkpoints/streetgeo_400m_mapillary/japan.pt --batch-size 32 --device cuda --amp bf16
```

已有下载目录可通过 `train --dataset-root ... --data-output ...` 自动准备后训练。`--download` 与 `--dataset-root`、直接清单和 `--resume` 互斥。重复 `--dataset-root` 只能合并 Mapillary 下载目录；同一图片 ID 会去重，坐标冲突会报错。

划分按采集序列和地理块整体分组，训练、验证、测试集之间排除近邻。需要至少三个独立分组；素材不足时先补充不同地点。输出是冻结的 `all.jsonl`、`train.jsonl`、`val.jsonl`、`test.jsonl` 与 `split_report.json`。相同数据与参数可复用校验通过的结果；图片或参数变化后使用新目录，避免改变活动训练的样本顺序。

`split --manifest ...` 与目录导入共用上述验证和冻结写入流程；省略 `--output-dir` 时生成到清单旁的 `splits/` 子目录，保留下载清单原文件。

每条清单须保留 `source: Mapillary`、数字字符串 `mapillary_id`、同 ID 的单张图片路径，以及经纬度。不要将其它来源的图片改标签冒充 Mapillary。原始下载目录和图片不复制、不删除。

默认新版模型为 397,937,418 参数，FP32 权重约 1.48 GiB，配置 `configs/streetgeo_400m.json`。建议 batch 32、BF16、CUDA，4 个读取进程，学习率 0.0003，预热 1,000 步。原版 134,686,730 参数使用 `configs/streetgeo_514m.json`，建议 batch 64。模型规模取决于可用硬件；容量检查不等于定位精度。不同规模需新建训练，权重维度不兼容。

## SigLIP2 NMPZ 单图训练

在 **模型架构** 选择 **SigLIP2 NMPZ（单图）**。每条 Mapillary 记录使用一张街景图。新模型先冻结 Google 的 SigLIP2 Base/16/224 视觉主干，只训练四查询池化、投影和定位头。建议从 CUDA、BF16、batch 8 开始。首次训练可自动下载固定版本的官方 `model.safetensors`，也可以在界面选择该文件。项目会优先读取 `checkpoints/pretrained/siglip2-base-patch16-224/model.safetensors`。

从工作区根目录启动训练：

```powershell
.venv\Scripts\python.exe -m streetgeo train --model-config configs/streetgeo_siglip2_nmpz.json --train-manifest data/world_mapillary/train.jsonl --val-manifest data/world_mapillary/val.jsonl --device cuda --amp bf16 --batch-size 8
```

预设使用硬标签。用 `--geo-smoothing 0.1` 和不同的 `--output` 可做轻度平滑对照，将联合分类目标的 10% 分配给按球面距离衰减的局部概率分布，衰减尺度为 250 km。若要在已有 NMPZ 权重上微调最后一层，可使用 `--init-from path/to/model.pt --siglip2-unfreeze-last 1 --visual-learning-rate 0.000005` 开始新阶段。完整续训使用同一训练任务的 `*.last.pt`。

在统一预测客户端选择训练输出的 `.pt` 即可离线预测。新版只接收一张图，保留四查询热力图。原架构的 checkpoint 继续使用其保存的图像预处理和视角数。分发包仅包含客户端程序和运行资源，训练及预测权重单独提供。

### 气候辅助监督

SigLIP2 NMPZ 可用柯本–盖革五大类（A–E）监督池化后的图像特征。在 CLI 加上 `--climate-loss-weight 0.1` 可启用，默认值为 `0`。给对照组使用权重 `0` 并指定独立输出文件。气候标签来自经纬度对应的栅格像素，低于 `--climate-min-confidence`（默认 80%）或缺失的标签只参与地理定位损失。

从 [Beck 等人的官方 NetCDF 归档](https://doi.org/10.6084/m9.figshare.21937568.v1) 提取 `1991_2020/koppen_geiger_0p01.nc`，保存到 `data/climate/beck2023_1991_2020/`，也可以用 `--climate-raster` 指定同一文件。该文件同时提供 `kg_class` 和 `kg_confidence`。训练日志会记录各类样本数和验证集气候指标，checkpoint 会记录栅格校验值及分类头，以便验证和续训。生成的 `.pt` 可由支持 SigLIP2 NMPZ 的推理代码直接加载。

## 环境专项微调：全图与底部梯形遮挡交替

此功能从已有 S.V.G.T. 模型开始一个新的微调阶段，训练图片仍只接受 Mapillary。原图按 EXIF 调整方向后，将底部居中的梯形填为均值色 `(124,116,104)`，再按现有方式缩放、填充和归一化；原始图片文件保留。横图、竖图均按原图内容计算范围，填充边不计入遮挡比例。

每一组图片按固定顺序完成两次优化器更新：**全图 → 专项遮挡**，然后进入下一组。两种输入共享同一个模型与原始坐标监督，比例固定为 1:1。每轮更新次数因此是普通训练的两倍，保存间隔、预热步数和日志中的 step 都按优化器更新次数计数。

使用与初始模型匹配的配置和独立的训练、验证清单。默认解冻末端两个视觉模块。每轮分别报告全图和遮挡模式的验证误差。

```powershell
python -m streetgeo train --environment-specialist --init-from checkpoints/current_model.pt --train-manifest data/world_mapillary/train.jsonl --val-manifest data/world_mapillary/val.jsonl --output checkpoints/environment/environment.pt --road-mask-top-width 0.20 --road-mask-bottom-width 1.0 --road-mask-height 0.45 --unfreeze-last 2 --learning-rate 0.00002 --visual-learning-rate 0.000005 --epochs 2 --warmup-steps 100 --batch-size 16 --device cuda --amp bf16
```

初始权重与验证清单为必填，训练和验证不能包含相同 Mapillary 图片。建议使用已有按序列、地理块隔离的准备结果。最佳模型以**全图验证中位误差**选取，并将初始模型纳入候选；若微调期间未改善此指标，最佳模型文件保留基线权重，最新微调进度仍在 `*.last.pt`。

完整断点保存训练策略、梯形尺寸、分组学习率、已冻结范围、当前 epoch/更新位置及下一阶段。全图更新完成后停止，恢复时从同一组图片的专项更新继续；专项更新完成后停止，恢复时进入下一组全图。继续时使用 `--resume ...environment.last.pt`，无需重新传初始权重或专项选项，续训会读取断点中的设置。

早期版本使用矩形下方 1/3 的完整断点，继续使用 `--resume` 时按原矩形策略恢复，避免混入不同训练条件。若要改为梯形，请将旧模型作为 `--init-from` 新建微调阶段；已保存的梯形断点续训时不能修改三个几何参数。

推理继续使用正常全图输入。固定梯形不是语义道路分割：它可能漏遮弯曲道路，或遮住道路外物体；实际定位效果需要用独立验证、测试结果判断。

## 中断与续训

训练每 500 批次或 300 秒保存完整状态，保留上一个状态。按一次 Ctrl+C，在安全批次边界保存。断电后从最近提交的 `*.last.pt` 恢复：

```powershell
python -m streetgeo train --resume checkpoints/streetgeo_400m_mapillary/best.last.pt --device cuda
```

精确续训沿用保存的架构、优化器、学习率、AMP、随机状态和批次位置，清单必须保持不变。OSV 或其它素材的续训状态被拒绝；已有推理权重仍可用于预测，或在架构兼容时通过 `--init-from` 初始化新的 Mapillary 阶段。

模型权重单独分发；任何分发压缩包不得包含权重、本机环境、缓存或个人偏好。
