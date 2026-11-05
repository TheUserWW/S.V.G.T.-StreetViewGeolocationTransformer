# S.V.G.T. Mapillary 预训练指南

预训练素材唯一来源为 **Mapillary**。OSV-5M、GSV-Cities、其它街景来源及混合素材均不再接受。训练集、验证集和续训都会逐条校验来源与原始 Mapillary 图片 ID。预测已有模型及单独的人工反馈学习流程保持各自入口。

## 图形工作台

运行 `scripts/windows/train.bat`，或执行：

```powershell
.venv\Scripts\python.exe -m streetgeo gui
```

1. **01 Mapillary 下载**：填写 Mapillary Client Access Token。Token 以密码形式显示，只保留在本次进程内，不写入本机偏好、下载数据库或日志。也可在启动前设置 `MAPILLARY_ACCESS_TOKEN` 环境变量。
2. 展开大洲，按 Ctrl/Shift 多选大洲、国家/地区或美国州。支持七大洲以及离线地图中的国家/地区；国家使用 ISO 三字母代码。选择多项时取并集，例如亚洲 + 法国表示亚洲及法国。
3. 对任意省、市、自定义区域，可选择 WGS84 GeoJSON（Polygon/MultiPolygon），或输入 `西经度 南纬度 东经度 北纬度`。边界与经纬度范围会进一步裁剪上面的选择；只用自定义范围时先清空国家和大洲。跨日期变更线的 bbox 分成两个下载任务。
4. 选择下载目录、目标总张数、每次查询页数、图片容量上限、预留磁盘容量、并发数、请求速率和分辨率。目标总张数包含目录中已下载的图片。
5. 点击 **下载并划分**。下载完成后自动划分并填入训练参数。点击 **仅下载 / 继续** 则只下载。随时可“停止并保存”，以后保持选区和质量参数不变继续下载。修改选区、质量参数或地理边界时使用新的下载目录。
6. **02 导入与划分** 也可以导入已下载目录（必须有 `all.jsonl` 和图片）。新增下载后使用新的划分输出目录。
7. **03 模型训练** 选择模型规模、参数与输出位置，点击“开始训练”。**04 预测地图** 打开开发者预测界面。

工作台自动保存选区、目录及普通参数；旧 OSV 工作台的数据路径会迁移为独立的 Mapillary 默认目录，旧文件不会删除。

左侧导航栏与主内容之间的分隔条可左右拖动，松开后自动记住宽度；下次打开恢复。缩小窗口时主内容会保留最小可用宽度。

## Token 与 API

从 [Mapillary 开发者门户](https://www.mapillary.com/dashboard/developers) 创建应用并取得 Client Access Token。参考 [官方 API 认证说明](https://mapillary.github.io/mapillary-python-sdk/docs/mapillary.models/mapillary.models.client/)。程序使用官方 `graph.mapillary.com` 查询图片，将 Token 放在 Authorization 请求头，只向 Mapillary API 发送；图片下载保留作者、来源与许可记录。

不需要把 Token 写进源代码或命令参数。命令行可用 `--prompt-token` 隐藏输入，或读取本机环境变量 `MAPILLARY_ACCESS_TOKEN`。

```powershell
# 查询可选国家/地区及内置行政区，不需要 Token
.venv\Scripts\python.exe -m streetgeo download --list-regions

# 按国家下载
.venv\Scripts\python.exe -m streetgeo download --prompt-token --countries CHN,JPN --output data/mapillary_east_asia --max-images 10000

# 按大洲下载；大洲代码：africa,asia,europe,north-america,south-america,oceania,antarctica
.venv\Scripts\python.exe -m streetgeo download --prompt-token --continents europe --output data/mapillary_europe

# 美国州或任意国家中的自定义区域
.venv\Scripts\python.exe -m streetgeo download --prompt-token --regions US-CA,US-NY --output data/mapillary_states
.venv\Scripts\python.exe -m streetgeo download --prompt-token --countries JPN --bbox 139.5 35.5 139.9 35.9 --output data/mapillary_tokyo
.venv\Scripts\python.exe -m streetgeo download --prompt-token --geojson data/region.geojson --output data/mapillary_region
```

下载器筛选 Mapillary 透视街景，跳过未投影的全景图和短边不足 224 像素的素材。优先下载指定分辨率，必要时尝试更大版本。下载质量、地理过滤与数据来源检查分别执行；实际可下载张数取决于 Mapillary 覆盖情况。

SQLite 保存图片 ID、已完成文件、分页游标、失败及延后重试任务。大范围先使用粗网格排队，查询时再细分。429 和临时错误有有界重试，过大查询区域自动细分。API 持续异常时安全暂停，不会自动开始训练。`--max-pages` 只限制本次查询数；已下载数据保留。`--max-gb` 是停止阈值，并发批次可能略微超出。

## 数据划分与训练

```powershell
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/mapillary_east_asia --output-dir data/world_mapillary
.venv\Scripts\python.exe -m streetgeo train --train-manifest data/world_mapillary/train.jsonl --val-manifest data/world_mapillary/val.jsonl --model-config configs/streetgeo_400m.json --output checkpoints/streetgeo_400m_mapillary/best.pt --epochs 4 --batch-size 32 --num-workers 4 --device cuda --amp bf16
```

也可一步完成下载、划分、训练：

```powershell
.venv\Scripts\python.exe -m streetgeo train --download --prompt-token --countries JPN --download-output data/mapillary_japan --data-output data/mapillary_japan_split --max-images 10000 --model-config configs/streetgeo_400m.json --output checkpoints/streetgeo_400m_mapillary/japan.pt --batch-size 32 --device cuda --amp bf16
```

已有下载目录可通过 `train --dataset-root ... --data-output ...` 自动准备后训练。`--download` 与 `--dataset-root`、直接清单和 `--resume` 互斥。重复 `--dataset-root` 只能合并 Mapillary 下载目录；同一图片 ID 会去重，坐标冲突会报错。

划分按采集序列和地理块整体分组，训练、验证、测试集之间排除近邻。需要至少三个独立分组；素材不足时先补充不同地点。输出是冻结的 `all.jsonl`、`train.jsonl`、`val.jsonl`、`test.jsonl` 与 `split_report.json`。相同数据与参数可复用校验通过的结果；图片或参数变化后使用新目录，避免改变活动训练的样本顺序。

`split --manifest ...` 与目录导入共用上述验证和冻结写入流程；省略 `--output-dir` 时生成到清单旁的 `splits/` 子目录，保留下载清单原文件。

每条清单须保留 `source: Mapillary`、数字字符串 `mapillary_id`、同 ID 的单张图片路径，以及经纬度。不要将其它来源的图片改标签冒充 Mapillary。原始下载目录和图片不复制、不删除。

默认新版模型为 397,937,418 参数，FP32 权重约 1.48 GiB，配置 `configs/streetgeo_400m.json`。建议 batch 32、BF16、CUDA，4 个读取进程，学习率 0.0003，预热 1,000 步。原版 134,686,730 参数使用 `configs/streetgeo_514m.json`，建议 batch 64。模型规模取决于可用硬件；容量检查不等于定位精度。不同规模需新建训练，权重维度不兼容。

## 环境专项微调：全图与底部梯形遮挡交替

此功能从已有 S.V.G.T. 模型开始一个新的微调阶段，训练图片仍只接受 Mapillary。原图按 EXIF 调整方向后，将底部居中的梯形填为均值色 `(124,116,104)`，再按现有方式缩放、填充和归一化；原始图片文件保留。横图、竖图均按原图内容计算范围，填充边不计入遮挡比例。

每一组图片按固定顺序完成两次优化器更新：**全图 → 专项遮挡**，然后进入下一组。两种输入共享同一个模型与原始坐标监督，比例固定为 1:1。每轮更新次数因此是普通训练的两倍，保存间隔、预热步数和日志中的 step 都按优化器更新次数计数。

在训练工作台的“模型训练”页：

1. 选择与初始模型一致的规模和配置，填写“新阶段初始权重”。
2. 选择独立的 Mapillary 训练、验证清单。
3. 将“环境专项微调”设为“开启 · 全图/专项交替”。工作台填入建议起点：2 轮、池化/融合/定位层学习率 `0.00002`、预热 100 步，并选择新的 `environment.pt` 输出位置。
4. 默认梯形为上底 **20% 图宽**、下底 **100% 图宽**、高 **45% 图高**，下底贴图片底部、左右居中。上边位于原图高度的 55%，顶点横坐标为原图宽度的 40% 和 60%。理论遮挡面积为整张图的 27%。这是抽看本机 16 个不同采集序列后选取的启发式起点，不是道路标注的统计平均；弯道和侧拍不能保证完整覆盖。点击“选择图片预览梯形遮挡”，可在对照窗口中观察范围，并修改工作台三个百分比实时更新。
5. 默认解冻最后 2 层视觉模块及视觉末端归一化层，视觉层学习率 `0.000005`；池化查询、池化注意力、朝向嵌入、融合层和定位头也参与训练。前面的视觉模块、切块层和位置编码冻结，冻结视觉模块的 dropout 关闭。层数须在所选模型架构范围内。
6. 启动后，程序先评估初始模型的全图和专项基线，再交替微调。每轮分别报告全图与专项的中位数、尾部距离误差和距离命中率。

命令行可直接沿用 checkpoint 的模型配置：

```powershell
.venv\Scripts\python.exe -m streetgeo train --environment-specialist --init-from checkpoints/current_model.pt --train-manifest data/world_mapillary/train.jsonl --val-manifest data/world_mapillary/val.jsonl --output checkpoints/environment/environment.pt --road-mask-top-width 0.20 --road-mask-bottom-width 1.0 --road-mask-height 0.45 --unfreeze-last 2 --learning-rate 0.00002 --visual-learning-rate 0.000005 --epochs 2 --warmup-steps 100 --batch-size 16 --device cuda --amp bf16
```

初始权重与验证清单为必填，训练和验证不能包含相同 Mapillary 图片。建议使用已有按序列、地理块隔离的准备结果。最佳模型以**全图验证中位误差**选取，并将初始模型纳入候选；若微调期间未改善此指标，最佳模型文件保留基线权重，最新微调进度仍在 `*.last.pt`。

完整断点保存训练策略、梯形尺寸、分组学习率、已冻结范围、当前 epoch/更新位置及下一阶段。全图更新完成后停止，恢复时从同一组图片的专项更新继续；专项更新完成后停止，恢复时进入下一组全图。继续时使用 `--resume ...environment.last.pt`，无需重新传初始权重或专项选项，工作台会锁定沿用的设置。

早期版本使用矩形下方 1/3 的完整断点，继续使用 `--resume` 时按原矩形策略恢复，避免混入不同训练条件。若要改为梯形，请将旧模型作为 `--init-from` 新建微调阶段；已保存的梯形断点续训时不能修改三个几何参数。

推理继续使用正常全图输入。固定梯形不是语义道路分割：它可能漏遮弯曲道路，或遮住道路外物体；实际定位效果需要用独立验证、测试结果判断。该模式与冻结整个视觉编码器的即时纠错学习分别运行，已有特征回放缓存不用于此次微调。

## 中断与续训

训练每 500 批次或 300 秒保存完整状态，保留上一个状态。点击停止或按一次 Ctrl+C，在安全批次边界保存。断电后从最近提交的 `*.last.pt` 恢复：

```powershell
.venv\Scripts\python.exe -m streetgeo train --resume checkpoints/streetgeo_400m_mapillary/best.last.pt --device cuda
```

精确续训沿用保存的架构、优化器、学习率、AMP、随机状态和批次位置，清单必须保持不变。OSV 或其它素材的续训状态被拒绝；已有推理权重仍可用于预测，或在架构兼容时通过 `--init-from` 初始化新的 Mapillary 阶段。

模型权重单独分发；任何分发压缩包不得包含权重、本机环境、缓存或个人偏好。
