"""Dependency-free Chinese/English text shared by explanation UI and exports."""

ENGLISH = {
    "预测解释": "Prediction explanation",
    "预测解释 · 池化注意力与遮挡实验": "Prediction explanation · Attention and occlusion",
    "蓝色较弱 → 红色较强；每幅图独立归一化。注意力是特征聚合线索，不是预测原因。":
        "Blue is weaker, red is stronger; each map is normalized separately. Attention shows feature pooling, not a causal explanation.",
    "视角": "View",
    "图片 {index}": "Image {index}",
    "查询平均": "Query average",
    "查询 {index}": "Query {index}",
    "叠加强度": "Overlay",
    "重新分析当前输入": "Analyze current inputs",
    "运行框选遮挡": "Run occlusion",
    "清除框选": "Clear selection",
    "导出 PNG": "Export PNG",
    "取消分析": "Cancel analysis",
    "原图 · 拖动框选遮挡区域": "Original · Drag to select an occlusion region",
    "注意力热力图": "Attention heatmap",
    "遮挡区域示意": "Occluded region preview",
    "等待分析…": "Waiting for analysis…",
    "正在准备模型…": "Preparing the model…",
    "正在准备分析…": "Preparing analysis…",
    "正在加载模型…": "Loading the model…",
    "正在全球范围内分析街景…": "Analyzing street views…",
    "正在官方街景覆盖国家／地区内分析…": "Analyzing within Google-covered countries/regions…",
    "正在提取各视角的池化注意力…": "Extracting pooling attention for each view…",
    "正在对选定区域进行遮挡预测…": "Predicting with the selected region occluded…",
    "正在导出固定布局的解释图…": "Exporting the explanation image…",
    "热力图已生成 · 当前图片：{path}": "Heatmap ready · Current image: {path}",
    "已框选新区域 · 点击“运行框选遮挡”；下方统计保留上次完成的实验":
        "Region selected. Run occlusion to update the comparison; previous results remain below.",
    "遮挡对比完成 · 分数固定比较原候选；原文件和模型参数未改动":
        "Occlusion complete · Comparing the same original cells; images and model weights are unchanged.",
    "导出预测解释": "Export prediction explanation",
    "分析未完成：{error}": "Analysis did not finish: {error}",
    "已导出：{path}": "Exported: {path}",
    "正在取消 · 等待当前模型加载或预测结束…": "Cancelling · Waiting for the current model load or prediction…",
    "分析已取消 · 已有结果保留": "Analysis cancelled · Previous results are preserved",
    "请先在测试端导入街景并选择模型": "Choose street-view images and a model first.",
    "模型：{path}": "Model: {path}",
    "Top-1：{lat:.6f}, {lon:.6f} · 网格概率 {prob:.4%}":
        "Top-1: {lat:.6f}, {lon:.6f} · Cell probability {prob:.4%}",
    "输入：{size} × {size} · 注意力网格 {side} × {side}":
        "Input: {size} × {size} · Attention grid: {side} × {side}",
    "四个查询没有预先规定的语义；不同查询的颜色不能直接比较绝对贡献。":
        "Queries have no predefined meaning. Colors across queries do not measure absolute contribution.",
    "请在左侧原图拖动框选，再运行遮挡；其余视角会一起保留用于预测。":
        "Drag a rectangle on the original, then run occlusion. Other views remain part of the prediction.",
    "遮挡图片 {index} · 灰色填充 · 其他视角保持原样":
        "Occluded image {index} · Gray fill · Other views unchanged",
    "原 Top-1：{lat:.6f}, {lon:.6f}": "Original Top-1: {lat:.6f}, {lon:.6f}",
    "遮挡后 Top-1：{lat:.6f}, {lon:.6f}": "Occluded Top-1: {lat:.6f}, {lon:.6f}",
    "预测位置移动：{distance:,.1f} km（不是与真实位置的误差）":
        "Prediction moved: {distance:,.1f} km (not the error against ground truth)",
    "固定比较原候选的网格概率（非校准正确率）：":
        "Fixed original candidate cell probabilities (not calibrated accuracy):",
    "候选 {index} / 网格 {cell}：{before:.4%} → {after:.4%}  变化 {change:+.4f} 个百分点":
        "Candidate {index} / Cell {cell}: {before:.4%} → {after:.4%}  Change {change:+.4f} percentage points",
    "遮挡会引入分布变化；分数下降或坐标移动不等于证明该区域是唯一原因。":
        "Occlusion changes the input distribution. A probability drop or location shift does not prove a unique cause.",
    "原图": "Original",
    "池化注意力 · {query}": "Pooling attention · {query}",
    "遮挡后输入区域示意": "Occluded input region preview",
    "蓝色较弱 → 红色较强；每幅热力图独立归一化。注意力反映特征聚合，不是预测原因。":
        "Blue is weaker, red is stronger; each map is normalized separately. Attention reflects feature pooling, not causation.",
    "原 Top-1：{lat:.6f}, {lon:.6f} · 网格概率 {prob:.4%}":
        "Original Top-1: {lat:.6f}, {lon:.6f} · Cell probability {prob:.4%}",
    "池化注意力：反映上下文特征的聚合位置；不是预测原因或正确率。":
        "Pooling attention: aggregation of contextual features, not a causal explanation or accuracy.",
}


def translate(source, language="zh", **values):
    return (ENGLISH.get(source, source) if language == "en" else source).format(**values)
