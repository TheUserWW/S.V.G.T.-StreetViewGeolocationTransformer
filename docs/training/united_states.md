# 美国 Mapillary 预训练

预训练只接受 Mapillary API 素材。使用 `--countries USA` 选择美国，或用 `--regions US-CA,US-NY` 选择州；也可使用自定义边界与经纬度范围。

```powershell
python -m streetgeo download --prompt-token --regions US-CA,US-NY --output data/mapillary_us --max-images 10000
python -m streetgeo prepare --dataset-root data/mapillary_us --output-dir data/mapillary_us_split
```

下载断点继续时保持相同选区与参数。新增数据后使用新的划分输出目录。完整流程、模型配置与恢复约束见 [Mapillary 训练指南](mapillary.md)。
