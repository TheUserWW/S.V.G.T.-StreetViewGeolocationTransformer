# Street-View Geolocation Transformer (S.V.G.T.)

## 中文

这是街景定位客户端：选择街景图片，在地图上查看模型预测位置，也可查看注意力热力图和运行手动遮挡分析。

1. 完整解压客户端文件夹，双击 `启动客户端.bat`（也可使用 `Launch Client.bat`）。
2. 首次启动需要联网安装依赖；之后图片与预测均在本机处理，地图可离线使用。
3. 在顶部点击“选择模型文件”，选择单独获取的 S.V.G.T. 模型（`.pt`、`.pth` 或 `.ckpt`）；也可直接输入路径。客户端会记住上次选择。随后选择 1–4 张同一地点的街景图。
4. 点击“预测位置”。地图上的编号对应左侧的候选坐标，点击候选可放大查看；可拖动、缩放地图，或返回全球总览。
5. 右上角可切换“中文 / English”，切换后保留当前图片与结果，并记住语言偏好。
6. 点击“关注区域 / 遮挡分析”，查看四个查询各自或平均的注意力热力图，调整叠加强度。在原图拖动框选，再点击“运行框选遮挡”，查看同一批原候选的概率变化和预测坐标变化；其他视角保留参与预测。可导出 PNG，也可取消分析。切换语言会同步更新已打开的解释窗口及统计文字，保留已有结果。
7. 可勾选“仅官方街景覆盖国家／地区”，按内置名单离线限制预测坐标，选择会自动记住；取消勾选恢复全球预测。名单包含部分覆盖地区，不保证具体道路有街景，也不是实时、完整的 Google 覆盖数据。关注区域和遮挡分析使用同一范围。切换范围会清除旧结果，需要重新预测。

拖动左侧栏与地图之间的分隔条可调整侧栏宽度，松开后自动保存；下次打开恢复。

热力图反映上下文特征的聚合位置；每幅图独立归一化，不能直接当作预测原因。遮挡造成的坐标移动不是与真实位置的误差。图片与模型权重均不会被修改。

**分发压缩包不包含模型权重；模型需要单独下载或传递。没有模型时也能正常打开客户端。**需要 Windows x64 与带 Tcl/Tk 的 64 位 Python 3.10–3.15；推荐 Python 3.12。推理使用 CPU，不需要 NVIDIA 显卡或训练数据。

## English

Choose street-view images, see predicted locations on a map, and inspect attention heatmaps or run manual occlusion experiments.

1. Extract the complete client folder and double-click `Launch Client.bat` (or `启动客户端.bat`).
2. Internet is needed for first-time dependency installation. Images and inference stay on your computer; the map works offline.
3. Select **Choose model file** at the top, then select your separately obtained S.V.G.T. model (`.pt`, `.pth` or `.ckpt`), or enter its path. The selected path is remembered. Next, choose 1–4 street views of the same location.
4. Select **Predict location**. Numbered map markers match the candidate coordinates on the left. Select a candidate to zoom in; pan, zoom or return to **World view**.
5. Switch between **English / 中文** in the upper-right corner. The current images and results are preserved, and the language preference is remembered.
6. Select **Attention / occlusion** to view individual pooling-query heatmaps or their average and adjust the overlay. Drag a rectangle on the original, then select **Run occlusion** to compare the same original candidate cells and the predicted coordinates. Other views remain part of the prediction. Export a PNG or cancel analysis. Language switching also updates the open explanation window and comparison text without losing results.
7. Enable **Google-covered countries/regions only** to restrict coordinates using the offline region allowlist. The preference is remembered; uncheck it for worldwide predictions. Partial-coverage regions are included. This is not a live or exhaustive Google coverage dataset and does not verify individual roads. Attention and occlusion use the same domain. Changing the domain clears previous results and requires a new prediction.

Drag the divider between the sidebar and map to adjust the sidebar width. The width is saved when you release it and restored next time.

Heatmaps show pooling of contextual features; each map is normalized separately and does not establish a prediction's cause. Location movement after occlusion is not error against ground truth. Images and model weights are never modified.

**The distribution archive does not include model weights. Download or share the model separately. The client opens normally without a model.** Windows x64 and 64-bit Python 3.10–3.15 with Tcl/Tk are required; Python 3.12 is recommended. Inference runs on the CPU. No NVIDIA GPU or training data is needed.

## 分发与更新 / Distribution and updates

Distribute the ZIP produced by `python -m streetgeo build-client` from the project root. This build command creates the client and archive automatically. The archive contains application files and offline maps only; it excludes all model weights, `.runtime`, caches and personal preferences. Distribute model files separately. Existing local model files are preserved during rebuilds but are never added to the ZIP.

使用构建命令自动生成的压缩包分发客户端；模型文件请单独发送。接收者首次运行后自行选择模型，程序会在本机记住路径和语言。重新打包不会删除本地已有权重，压缩包不会包含它们。

The distributed UI includes model-file selection, street-view image selection, map predictions, attention heatmaps, manual occlusion and explanation PNG export. Training, device settings, manual coordinates, full-dataset evaluation and screenshot calibration remain outside this UI.


## 故障排查 / Troubleshooting

- 安装被中断：重新运行启动器。 / Interrupted installation: run the launcher again.
- 启动失败：查看 `.runtime/startup.log`。 / Startup failure: inspect `.runtime/startup.log`.
- 预测失败：查看 `.runtime/prediction.log`，确认图片可读且所选模型完整。 / Prediction failure: inspect `.runtime/prediction.log` and check the images and selected model.
- 缺少 tkinter：在 Python 安装器中安装 Tcl/Tk and IDLE。 / Missing tkinter: add Tcl/Tk and IDLE using the Python installer.
- `DLL load failed`：检查 Microsoft Visual C++ 运行库。 / Check the Microsoft Visual C++ runtime.

Developers can use `python bootstrap.py --setup-only` to prepare dependencies and `python bootstrap.py --self-test` to check startup without model weights. Add `--checkpoint "path/to/model.pt"` to run an actual CPU prediction check against an external model.

Map boundaries: UN Geospatial / OCHA; source details are retained in `assets/maps/un_map_sources.json`. Map display does not imply UN endorsement of territorial status. US state boundaries: Census TIGERweb. Original third-party notices remain bundled.
