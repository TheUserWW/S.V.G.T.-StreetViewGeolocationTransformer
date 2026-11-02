"""Attention inspection and fixed-candidate occlusion experiments.

No hooks or analysis state are installed permanently. Raw pooling attention is
an information-routing diagnostic, not a causal explanation of the prediction.
"""

from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, PngImagePlugin
import torch
from torch.nn import functional as F

from streetgeo.datasets.images import IMAGE_MEAN, IMAGE_STD, load_image
from streetgeo.inference.backend import read_picture
from streetgeo.inference.explanation_text import translate
from streetgeo.paths import ROOT


class AnalysisCancelled(Exception):
    pass


def check_cancelled(cancelled):
    if cancelled():
        raise AnalysisCancelled("已取消分析")


def content_box(picture, size):
    """Match Pillow ImageOps.pad's actual integer resize and centered padding."""
    contained = ImageOps.contain(picture, (size, size))
    width, height = contained.size
    if width == 0 or height == 0:
        raise ValueError("图片长宽比过于极端，缩放后没有有效像素")
    left = round((size - width) / 2)
    top = round((size - height) / 2)
    return left, top, left + width, top + height


def candidates(output):
    return [{"latitude": float(lat), "longitude": float(lon),
             "cell_probability": float(prob), "cell_id": int(cell)}
            for (lat, lon), prob, cell in zip(output.candidate_coordinates[0].cpu().tolist(),
                                             output.candidate_probabilities[0].cpu().tolist(),
                                             output.candidate_cell_ids[0].cpu().tolist())]


def joint_probability(output, cell_id, children):
    parent, child = divmod(cell_id, children)
    return float((F.log_softmax(output.coarse_logits.float(), -1)[0, parent]
                  + F.log_softmax(output.fine_logits.float(), -1)[0, parent, child]).exp())


def analyze_attention(model, image_paths, *, google_coverage_only=False, cancelled=lambda: False,
                      progress=lambda text: None):
    """Return Q maps per view, measured on the same forward as the predictions.

The model's forward stays unchanged: a temporary hook recomputes only the
pooling Q/K weights (averaged over heads) from its actual inputs. In particular,
the optimized self-attention path and the checkpoint format stay untouched.
"""
    if not 1 <= len(image_paths) <= model.config.max_views:
        raise ValueError("请选择同一地点的有效数量街景")
    pictures, tensors = [], []
    for path in image_paths:
        check_cancelled(cancelled)
        pictures.append(read_picture(path))
        tensors.append(load_image(path, model.config.image_size))
    inputs = torch.stack(tensors).unsqueeze(0)
    captured = []

    def capture(module, args, output):
        query, key, _ = args
        width = module.embed_dim
        # This project's pool uses equal Q/K/V dimensions, batch_first=True.
        bias = module.in_proj_bias
        q = F.linear(query, module.in_proj_weight[:width], None if bias is None else bias[:width])
        k = F.linear(key, module.in_proj_weight[width:2 * width],
                     None if bias is None else bias[width:2 * width])
        heads, head_width = module.num_heads, width // module.num_heads
        q = q.reshape(q.shape[0], q.shape[1], heads, head_width).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], heads, head_width).transpose(1, 2)
        weights = (q.float() @ k.float().transpose(-2, -1) / math.sqrt(head_width)).softmax(-1)
        captured.append(weights.mean(1).cpu())

    was_training = model.training
    hook = model.encoder.pool_attention.register_forward_hook(capture)
    try:
        model.eval()
        progress("正在提取各视角的池化注意力…")
        check_cancelled(cancelled)
        options = {}
        if google_coverage_only:
            from streetgeo.inference.coverage import google_coverage_mask
            options["coordinate_filter"] = google_coverage_mask
        with torch.inference_mode():
            output = model(inputs.to(next(model.parameters()).device),
                           top_k=min(5, model.config.total_cells), **options)
        check_cancelled(cancelled)
    finally:
        hook.remove()
        model.train(was_training)
    side = model.config.image_size // model.config.patch_size
    maps = captured[0].reshape(len(pictures), model.config.view_queries, side, side).numpy()
    if not np.isfinite(maps).all():
        raise ValueError("注意力中出现非有限数值")
    return {"paths": [str(Path(path).resolve()) for path in image_paths],
            "pictures": pictures, "inputs": inputs, "maps": maps,
            "image_size": model.config.image_size, "candidates": candidates(output),
            "google_coverage_only": bool(google_coverage_only),
            "note": "池化注意力：反映上下文特征的聚合位置；不是预测原因或正确率。"}


def occlude_region(model, analysis, view, rectangle, *, cancelled=lambda: False,
                   progress=lambda text: None):
    """Replace one selected view region, preserving all other views.

    rectangle is in oriented original-image normalized coordinates. The actual
    replaced pixels are quantized to the model input grid, then mapped back for
    display. Candidate probabilities are compared for the ORIGINAL five cells.
    """
    check_cancelled(cancelled)
    if type(view) is not int or not 0 <= view < len(analysis["pictures"]):
        raise ValueError("无效图片序号")
    if len(rectangle) != 4 or not all(math.isfinite(float(v)) for v in rectangle):
        raise ValueError("无效遮挡区域")
    x0, y0, x1, y1 = map(float, rectangle)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("请在原图内拖动选择一个非空矩形")
    size = analysis["image_size"]
    left, top, right, bottom = content_box(analysis["pictures"][view], size)
    width, height = right - left, bottom - top
    box = (max(left, math.floor(left + x0 * width)), max(top, math.floor(top + y0 * height)),
           min(right, math.ceil(left + x1 * width)), min(bottom, math.ceil(top + y1 * height)))
    x0, y0, x1, y1 = box
    inputs = analysis["inputs"].clone()
    fill = inputs.new_tensor([(channel / 255 - mean) / std
                              for channel, mean, std in zip((124, 116, 104), IMAGE_MEAN, IMAGE_STD)])
    inputs[0, view, :, y0:y1, x0:x1] = fill[:, None, None]
    was_training = model.training
    try:
        model.eval()
        progress("正在对选定区域进行遮挡预测…")
        options = {}
        if analysis.get("google_coverage_only"):
            from streetgeo.inference.coverage import google_coverage_mask
            options["coordinate_filter"] = google_coverage_mask
        with torch.inference_mode():
            output = model(inputs.to(next(model.parameters()).device),
                           top_k=min(5, model.config.total_cells), **options)
            comparisons = [{"cell_id": item["cell_id"], "before": item["cell_probability"],
                            "after": joint_probability(output, item["cell_id"], model.config.children_per_cell)}
                           for item in analysis["candidates"]]
        check_cancelled(cancelled)
    finally:
        model.train(was_training)
    effective = ((x0 - left) / width, (y0 - top) / height,
                 (x1 - left) / width, (y1 - top) / height)
    picture = analysis["pictures"][view].copy()
    pw, ph = picture.size
    pixels = (round(effective[0] * pw), round(effective[1] * ph),
              round(effective[2] * pw), round(effective[3] * ph))
    picture.paste((124, 116, 104), pixels)
    return {"view": view, "rectangle": effective, "input_box": box, "picture": picture,
            "candidates": candidates(output), "comparisons": comparisons}


def heatmap_image(analysis, view, query=None, opacity=0.5):
    """Crop model letterbox padding before projecting the map to the original."""
    opacity = float(opacity)
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("透明度必须在 0～1 之间")
    picture = analysis["pictures"][view]
    maps = analysis["maps"][view]
    values = maps.mean(0) if query is None else maps[query]
    low, high = float(values.min()), float(values.max())
    values = np.zeros_like(values) if high - low < 1e-12 else (values - low) / (high - low)
    small = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255))
    size = analysis["image_size"]
    field = small.resize((size, size), Image.Resampling.BILINEAR)
    field = field.crop(content_box(picture, size)).resize(picture.size, Image.Resampling.BILINEAR)
    values = np.asarray(field, dtype=np.float32) / 255
    # Blue -> cyan -> yellow -> red, independent relative scaling for each map.
    colors = np.array([[35, 63, 150], [35, 193, 188], [253, 220, 68], [224, 55, 43]])
    rgb = np.stack([np.interp(values, np.linspace(0, 1, len(colors)), colors[:, i])
                    for i in range(3)], axis=-1).astype(np.uint8)
    return Image.blend(picture, Image.fromarray(rgb), opacity)


def comparison_text(analysis, occlusion, language="zh"):
    def t(source, **values):
        return translate(source, language, **values)
    before, after = analysis["candidates"][0], occlusion["candidates"][0]
    lat1, lat2 = math.radians(before["latitude"]), math.radians(after["latitude"])
    delta = math.radians(after["longitude"] - before["longitude"])
    hav = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta / 2) ** 2
    distance = 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, hav))))
    lines = [t("遮挡图片 {index} · 灰色填充 · 其他视角保持原样", index=occlusion["view"] + 1),
             t("原 Top-1：{lat:.6f}, {lon:.6f}", lat=before["latitude"], lon=before["longitude"]),
             t("遮挡后 Top-1：{lat:.6f}, {lon:.6f}", lat=after["latitude"], lon=after["longitude"]),
             t("预测位置移动：{distance:,.1f} km（不是与真实位置的误差）", distance=distance),
             t("固定比较原候选的网格概率（非校准正确率）：")]
    for index, item in enumerate(occlusion["comparisons"], 1):
        lines.append(t("候选 {index} / 网格 {cell}：{before:.4%} → {after:.4%}  变化 {change:+.4f} 个百分点",
                       index=index, cell=item["cell_id"], before=item["before"], after=item["after"],
                       change=(item["after"] - item["before"]) * 100))
    lines.append(t("遮挡会引入分布变化；分数下降或坐标移动不等于证明该区域是唯一原因。"))
    return "\n".join(lines)


def export_explanation(analysis, view, query, opacity, path, occlusion=None, *, language="zh"):
    """Save a fixed readable layout independent of the current window size."""
    def t(source, **values):
        return translate(source, language, **values)
    if occlusion is not None and occlusion["view"] != view:
        occlusion = None
    def font(size):
        for name in ("C:/Windows/Fonts/msyh.ttc", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    heading, body = font(28), font(19)
    original = analysis["pictures"][view].copy()
    query_name = t("查询平均") if query is None else t("查询 {index}", index=query + 1)
    panels = [(t("原图"), original), (t("池化注意力 · {query}", query=query_name),
                                     heatmap_image(analysis, view, query, opacity))]
    if occlusion is not None and occlusion["view"] == view:
        w, h = original.size
        ImageDraw.Draw(original).rectangle(tuple(round(v * scale) for v, scale in
                                                 zip(occlusion["rectangle"], (w, h, w, h))),
                                          outline="#efbb73", width=max(2, w // 300))
        panels.append((t("遮挡后输入区域示意"), occlusion["picture"]))
    width = 1500
    card_width = (width - 40) // len(panels)
    note = ["S.V.G.T. · " + t("预测解释"), Path(analysis["paths"][view]).name,
            t("蓝色较弱 → 红色较强；每幅热力图独立归一化。注意力反映特征聚合，不是预测原因。"),
            t("模型：{path}", path=str(analysis.get("checkpoint", "")))]
    first = analysis["candidates"][0]
    note.append(t("原 Top-1：{lat:.6f}, {lon:.6f} · 网格概率 {prob:.4%}", lat=first["latitude"],
                  lon=first["longitude"], prob=first["cell_probability"]))
    if occlusion is not None:
        note.extend(comparison_text(analysis, occlusion, language).splitlines())
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = []
    for line in note[1:]:
        part = ""
        for char in line:
            if measure.textlength(part + char, font=body) > width - 40:
                wrapped.append(part)
                part = ""
            part += char
        wrapped.append(part)
    image = Image.new("RGB", (width, 626 + 32 * len(wrapped)), "#10161e")
    draw = ImageDraw.Draw(image)
    draw.text((20, 14), note[0], font=heading, fill="#e3eaf2")
    for index, (title, picture) in enumerate(panels):
        draw.text((20 + index * card_width, 66), title, font=body, fill="#65d6c2")
        shown = ImageOps.contain(picture, (card_width - 16, 480), Image.Resampling.LANCZOS)
        image.paste(shown, (20 + index * card_width + (card_width - 16 - shown.width) // 2,
                           105 + (480 - shown.height) // 2))
    y = 610
    for line in wrapped:
        draw.text((20, y), line, font=body, fill="#e3eaf2")
        y += 32
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", t(analysis["note"]))
    image.crop((0, 0, width, y + 16)).save(path, format="PNG", pnginfo=metadata)


def run_explanation_job(engine, action, payload, *, checkpoint, images, device="cpu",
                        analysis=None, occlusion=None, language="zh", google_coverage_only=False,
                        cancelled=lambda: False, progress=lambda text: None):
    """Single-worker operation shared by developer and portable prediction apps."""
    check_cancelled(cancelled)
    if action == "attention":
        result = engine.predict(checkpoint, images, device=device, progress=progress,
                                google_coverage_only=google_coverage_only)
        check_cancelled(cancelled)
        data = analyze_attention(engine.model, images, google_coverage_only=google_coverage_only,
                                 cancelled=cancelled, progress=progress)
        data.update(checkpoint=result["checkpoint"], model_key=engine.key)
    elif action == "occlusion":
        key = analysis["model_key"]
        selected = Path(checkpoint).expanduser()
        selected = selected if selected.is_absolute() else ROOT / selected
        if (str(selected.resolve()) != key[0] or device != key[3]
                or bool(analysis.get("google_coverage_only")) != bool(google_coverage_only)
                or [str(Path(image).resolve()) for image in images] != analysis["paths"]):
            raise ValueError("输入图片、模型或设备选择已变化，请先重新分析当前输入")
        stamp = Path(key[0]).stat()
        if (engine.key != key or engine.model is None
                or (stamp.st_mtime_ns, stamp.st_size) != key[1:3]):
            raise ValueError("模型已切换或 checkpoint 已更新，请先重新分析当前输入")
        data = occlude_region(engine.model, analysis, *payload, cancelled=cancelled, progress=progress)
    elif action == "export":
        view, query, opacity, path = payload
        progress("正在导出固定布局的解释图…")
        export_explanation(analysis, view, query, opacity, path, occlusion, language=language)
        data = path
    else:
        raise ValueError("未知解释操作")
    check_cancelled(cancelled)
    return data
