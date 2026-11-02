# Google 官方街景覆盖国家／地区筛选

本地预测地图和中英双语分发客户端均提供可选开关，默认关闭，并记住用户的选择。启用后，仅返回位于内置国家／地区名单边界内的模型预测坐标；地图当前显示范围不参与筛选。不需要联网、API Key 或重新训练。

命令行也可使用：

```powershell
python -m streetgeo predict photo.jpg --checkpoint path/to/model.pt --google-coverage-only --json
```

这是**国家／地区级粗筛**，包含部分覆盖地区。内置名单不是 Google 官方发布的覆盖数据，不是完整或实时的道路覆盖图；名单内的国家仍可能存在大片没有街景的区域。该选项不会将坐标吸附到街景道路，也不验证某个全景点是否存在。精确位置查询可参考 [Google Street View API 的来源筛选](https://developers.google.com/maps/documentation/streetview/request-streetview)，其中 `source=google` 表示 Google 官方采集内容。

名单及版本保存在 `assets/maps/google_streetview_regions.json`，边界沿用项目已有的 UN Geospatial / OCHA `un_countries.geojson`，按 ISO 3166-1 alpha-3 代码匹配；香港、澳门等地区使用各自边界。边界比例尺为 1:25,000,000，部分海岸、小岛和狭小地区会有遗漏或偏差。范围外国家和多边形内部空洞会被排除。覆盖状态变化时应维护名单、版本和来源链接，再重新构建客户端。

模型按原始联合概率依次扫描候选坐标，跳过范围外候选，直到找到最多五个结果或检查完全部网格。分数保留全球原始概率，不重新归一化，不代表覆盖条件下的准确率。没有符合范围的预测坐标时会报错，不会回退为全球预测或制造替代坐标。小网格模型可能返回少于五个候选。

单次预测、随机 test、全量 test 评估、关注区域和遮挡分析使用同一开关。真实测试标签不参与筛选；全量评估仍遍历原测试集，范围外样本也计入误差。评估报告和导出的预测 JSON 记录是否启用、名单版本及粗筛级别，便于区分实验。切换开关会清除旧预测并关闭旧解释窗口。

重新构建分发客户端：`python -m streetgeo build-client --output clinet`。构建会带入范围代码与名单；分发 ZIP 不包含权重、本地环境、缓存或个人偏好。
