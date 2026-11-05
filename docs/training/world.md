# 全球 Mapillary 预训练

唯一素材来源为 Mapillary。使用工作台多选大洲、国家/地区；行政区域可用内置美国州或任意 WGS84 GeoJSON/bbox。图像始终来自 Mapillary，离线地理多边形仅用于范围过滤。

```powershell
.venv\Scripts\python.exe -m streetgeo download --prompt-token --continents asia,europe --output data/mapillary_world --max-images 10000
.venv\Scripts\python.exe -m streetgeo prepare --dataset-root data/mapillary_world --output-dir data/mapillary_world_split
```

每次下载目标包含已经完成的图片。选择更大范围不保证有足够 Mapillary 覆盖。建议先用较小区域验证数据与模型，再扩大范围；保持采集序列和地理位置隔离，使用独立验证与测试数据评价真实定位误差。OSV、GSV-Cities、其它图片来源和混合素材不接受预训练或续训。

完整流程见 [Mapillary 训练指南](mapillary.md)。
