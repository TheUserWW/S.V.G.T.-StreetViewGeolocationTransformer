"""Offline statistical figures for the developer evaluation reports only."""

import csv
import json
import math
import os
from pathlib import Path
import tempfile
import threading

_PLOT_LOCK = threading.RLock()
CHART_VIEWS = (("overview", "统计总览"), ("countries", "国家与覆盖情况"))
COLORS = {"background": "#10161e", "panel": "#18222e", "text": "#e3eaf2", "muted": "#a1b0c1",
          "grid": "#334255", "accent": "#65d6c2", "mean": "#efbb73", "median": "#8ab8ff"}


def distribution_data(errors, scores):
    """Exact histogram counts and a bounded, deterministic empirical CDF.

    Integer scores are binned [0,500), ... [4500,5000]. The CDF is exact
    for <=512 distinct distances; larger curves retain sampled true ranks.
    """
    if len(errors) != len(scores):
        raise ValueError("评分和距离样本数量不一致")
    counts = [0] * 10
    for error, score in zip(errors, scores):
        if (not math.isfinite(error) or error < 0 or not math.isfinite(score)
                or not 0 <= score <= 5000 or score != int(score)):
            raise ValueError("统计结果含无效距离或非整数评分")
        counts[min(int(score) // 500, 9)] += 1
    ordered = sorted(errors)
    # Deduplicate before downsampling so a tied value has its correct ECDF
    # mass, including when all predictions have the same error.
    points = []
    for rank, error in enumerate(ordered, 1):
        if rank == len(ordered) or ordered[rank] != error:
            points.append((error, rank / len(ordered)))
    if len(points) > 512:
        points = [points[round(i * (len(points) - 1) / 511)] for i in range(512)]
    return {"sample_count": len(errors), "score_bin_edges": list(range(0, 5001, 500)),
            "score_bin_counts": counts,
            "error_cdf": {"error_km": [p[0] for p in points], "fraction": [p[1] for p in points],
                          "max_points": 512, "downsampled": len(set(errors)) > 512}}


def load_evaluation_report(path):
    """Open old reports without inference; rebuild chart aggregates from CSV.

    Resolve files beside the chosen summary, supporting moved report folders.
    Never trust a saved absolute output_directory to choose where to write.
    """
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"top1", "successful", "status", "by_country", "coverage", "checkpoint", "test_directory",
                "batch_size", "failed", "not_run", "elapsed_seconds", "full_test_complete", "best_of_5_error_km"}
    if not isinstance(report, dict) or not required <= report.keys():
        raise ValueError("请选择全量 test 评估生成的 summary.json")
    if report["status"] not in ("completed", "cancelled", "failed"):
        raise ValueError("这份报告尚未完成保存，请在评估结束或停止后打开")
    report["output_directory"] = str(path.parent)
    if "visualization" not in report:
        errors, scores = [], []
        results = path.parent / "results.csv"
        if results.is_file():
            with results.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("status") == "ok":
                        errors.append(float(row["top1_error_km"]))
                        scores.append(float(row["top1_score"]))
            if len(errors) != report["successful"]:
                raise ValueError("results.csv 与 summary.json 的成功样本数量不一致，请选择同一轮评估的报告")
            report["visualization"] = distribution_data(errors, scores)
        elif not report["successful"]:
            report["visualization"] = distribution_data([], [])
    return report


def _number(value, suffix=""):
    return "—" if value is None else f"{value:,.1f}{suffix}"


def _heading(figure, report, view):
    partial = not report.get("full_test_complete", False)
    scope = "部分结果" if partial else "完整测试集"
    states = {"completed": "遍历完成", "cancelled": "已停止", "failed": "评估失败"}
    figure.suptitle(f"S.V.G.T. · {dict(CHART_VIEWS)[view]}  |  {scope} · {states.get(report['status'], report['status'])}",
                   x=.055, y=.975, ha="left", fontsize=19, weight="bold", color=COLORS["text"])
    score, error = report["top1"]["score"], report["top1"]["error_km"]
    figure.text(.055, .923, f"平均分  {_number(score['mean'])}     中位分  {_number(score['median'])}"
                f"     平均误差  {_number(error['mean'], ' km')}     中位误差  {_number(error['median'], ' km')}",
                fontsize=12, color=COLORS["text"])
    total = (report.get("coverage") or {}).get("available_samples")
    model = Path(report.get("checkpoint", "")).name
    model = model if len(model) <= 28 else model[:25] + "…"
    total_text = f"{total:,}" if total is not None else "未知"
    figure.text(.055, .885, f"{model}  ·  成功 {report['successful']:,} / 本地可测 {total_text}"
                "  ·  评分/误差仅统计成功样本；缺失、失败与未运行不计分", fontsize=10, color=COLORS["muted"])
    figure.text(.055, .022, "Top-1 · GeoGuessr World 近似评分 / 5000 · 误差单位 km  |  "
                + report.get("created_at", "")[:19].replace("T", " ") + " UTC", fontsize=9, color=COLORS["muted"])


def _style_axis(axis, title, *, horizontal=False):
    axis.set_facecolor(COLORS["panel"])
    axis.set_title(title, loc="left", color=COLORS["text"], fontsize=12, pad=15)
    axis.tick_params(colors=COLORS["muted"], labelsize=9)
    for name, spine in axis.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(COLORS["grid"])
    axis.grid(axis="x" if horizontal else "y", color=COLORS["grid"], linewidth=.6, alpha=.6)
    axis.set_axisbelow(True)
    axis.xaxis.label.set_color(COLORS["muted"])
    axis.yaxis.label.set_color(COLORS["muted"])


def _vertical_bars(axis, labels, values, *, percent=False):
    bars = axis.bar(range(len(values)), values, color=COLORS["accent"], width=.65)
    axis.set_xticks(range(len(values)), labels)
    axis.set_ylim(0, 110 if percent else max(1, max(values, default=0) * 1.23))
    axis.bar_label(bars, labels=[f"{v:.1f}%" if percent else f"{v:,}" for v in values],
                   padding=5, color=COLORS["text"], fontsize=9)
    axis.set_ylabel("成功样本占比 / %" if percent else "样本数")


def build_evaluation_figure(report, view="overview"):
    """Create a Matplotlib Figure without importing pyplot or starting Tk."""
    if view not in dict(CHART_VIEWS):
        raise ValueError("未知统计图视图")
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    families = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((font for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
                    if font in families), "DejaVu Sans")
    with _PLOT_LOCK, matplotlib.rc_context({"font.family": chinese, "axes.unicode_minus": False,
                                          "text.parse_math": False, "svg.fonttype": "path"}):
        figure = Figure(figsize=(13, 9), dpi=100, facecolor=COLORS["background"])
        _heading(figure, report, view)
        if view == "overview":
            axes = figure.subplots(2, 2)
            figure.subplots_adjust(left=.075, right=.965, top=.79, bottom=.115, hspace=.58, wspace=.26)
            score_axis, error_axis, hit_axis, cdf_axis = axes.flat
            _style_axis(score_axis, "评分分布 · 越高越好")
            data = report.get("visualization")
            if data is not None:
                edges, counts = data["score_bin_edges"], data["score_bin_counts"]
                bars = score_axis.bar(edges[:-1], counts, width=450, align="edge", color=COLORS["accent"])
                score_axis.set_xlim(-100, 5100)
                score_axis.set_xticks(list(range(0, 5001, 1000)))
                score_axis.set_ylim(0, max(1, max(counts, default=0) * 1.25))
                score_axis.bar_label(bars, labels=[f"{n:,}" if n else "" for n in counts], padding=4,
                                      fontsize=8, color=COLORS["text"])
                if report["successful"]:
                    for key, title, color, style in (("mean", "平均", "mean", "--"), ("median", "中位", "median", ":")):
                        score_axis.axvline(report["top1"]["score"][key], color=COLORS[color], linestyle=style,
                                           linewidth=1.7, label=f"{title} {_number(report['top1']['score'][key])}")
                    score_axis.legend(fontsize=9, facecolor=COLORS["panel"], labelcolor=COLORS["text"], edgecolor="none")
            else:
                score_axis.text(.5, .5, "历史报告缺少 results.csv，无法恢复评分分布", transform=score_axis.transAxes,
                                ha="center", color=COLORS["muted"])
            score_axis.set_xlabel("单图评分 / 5000（每段 500 分，最后一段含 5000）")
            score_axis.set_ylabel("样本数")
            score_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            _style_axis(error_axis, "距离误差分布 · 越低越好")
            bins = report["top1"]["distance_bins_km"]
            _vertical_bars(error_axis, list(bins), list(bins.values()))
            error_axis.set_xlabel("误差区间 / km（第一段含 0，其余左开右闭）")
            _style_axis(hit_axis, "距离阈值命中率 · 累计占比")
            within = report["top1"]["within_km"]
            _vertical_bars(hit_axis, [f"≤{k}" for k in within],
                           [100 * (v["fraction"] or 0) for v in within.values()], percent=True)
            hit_axis.set_xlabel("距离阈值 / km")
            _style_axis(cdf_axis, "距离误差累计曲线 · 越靠左上越好")
            if data and data["error_cdf"]["error_km"]:
                curve = data["error_cdf"]
                x, y = curve["error_km"], [v * 100 for v in curve["fraction"]]
                cdf_axis.step([0, *x], [0, *y], where="post", color=COLORS["accent"], linewidth=1.7)
                median = report["top1"]["error_km"]["median"]
                cdf_axis.axvline(median, color=COLORS["median"], linestyle=":", linewidth=1.4,
                                 label=f"中位误差 {_number(median, ' km')}")
                cdf_axis.legend(fontsize=9, facecolor=COLORS["panel"], labelcolor=COLORS["text"], edgecolor="none")
                cdf_axis.set_xscale("symlog", linthresh=1)
                cdf_axis.set_xlim(0, max(1, x[-1] * 1.1))
                cdf_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:,.0f}"))
            else:
                cdf_axis.text(.5, .5, "暂无逐图距离数据", transform=cdf_axis.transAxes, ha="center", color=COLORS["muted"])
            cdf_axis.set_ylim(0, 105)
            cdf_axis.set_xlabel("距离误差 / km（1 km 以上对数刻度，曲线最多 512 点）")
            cdf_axis.set_ylabel("累计成功样本 / %")
        else:
            axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
            figure.subplots_adjust(left=.12, right=.92, top=.79, bottom=.13, wspace=.42)
            country_axis, coverage_axis = axes
            _style_axis(country_axis, "国家/地区评分 · 按样本数选前 15 组", horizontal=True)
            groups = sorted(report["by_country"].items(), key=lambda item: (-item[1]["score"]["count"], item[0]))[:15]
            if groups:
                labels = [f"{'未知' if name == '(unknown)' else name}  (n={group['score']['count']:,})" for name, group in groups]
                country_axis.barh(range(len(groups)), [g["score"]["mean"] for _, g in groups],
                                  color=COLORS["accent"], height=.58, label="平均分")
                country_axis.scatter([g["score"]["median"] for _, g in groups], range(len(groups)),
                                     marker="D", color=COLORS["mean"], s=24, label="中位分", zorder=3)
                country_axis.set_yticks(range(len(groups)), labels)
                country_axis.invert_yaxis()
                country_axis.legend(loc="lower right", fontsize=9, facecolor=COLORS["panel"],
                                    labelcolor=COLORS["text"], edgecolor="none")
            else:
                country_axis.text(.5, .5, "暂无成功样本", transform=country_axis.transAxes, ha="center", color=COLORS["muted"])
            country_axis.set_xlim(0, 5000)
            country_axis.set_xlabel("评分 / 5000 · 全部组别见详细统计")
            _style_axis(coverage_axis, "测试覆盖与排除数量", horizontal=True)
            coverage = report.get("coverage") or {}
            labels = ["成功评估", "图片失败", "未运行", "缺失图片", "无效元数据", "重复 ID"]
            values = [report["successful"], report["failed"], report["not_run"], coverage.get("missing_image"),
                      coverage.get("invalid_id_or_coordinates"), coverage.get("duplicate_id")]
            bars = coverage_axis.barh(range(len(labels)), [v or 0 for v in values],
                                      color=[COLORS["accent"], *([COLORS["mean"]] * 5)], height=.55)
            coverage_axis.set_yticks(range(len(labels)), labels)
            coverage_axis.invert_yaxis()
            coverage_axis.bar_label(bars, labels=[f"{v:,}" if v is not None else "未知" for v in values],
                                     padding=5, fontsize=10, color=COLORS["text"])
            coverage_axis.set_xlim(0, max(1, max((v or 0 for v in values), default=0) * 1.3))
            coverage_axis.set_xlabel("数量 · 重复 ID 按 CSV 多余行计")
        return figure


def save_evaluation_charts(report):
    """Atomically export standalone PNG/SVG figures, keeping inference optional."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    folder = Path(report["output_directory"])
    saved = {}
    with _PLOT_LOCK:
        for view, _ in CHART_VIEWS:
            figure = build_evaluation_figure(report, view)
            FigureCanvasAgg(figure)
            try:
                for suffix in ("png", "svg"):
                    name = f"statistics_{view}.{suffix}"
                    descriptor, temporary = tempfile.mkstemp(prefix=name, suffix=".tmp", dir=folder)
                    os.close(descriptor)
                    try:
                        figure.savefig(temporary, format=suffix, dpi=160, facecolor=figure.get_facecolor())
                        os.replace(temporary, folder / name)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                    saved[f"{view}_{suffix}"] = name
            finally:
                figure.clear()
    return saved


def export_evaluation_chart(report, view, path, *, limits=None):
    """Export a fresh, fixed-size figure while preserving the user's data zoom.

    Tk resizes its live figure in place. Exporting that figure directly would
    bake a narrow/short window's geometry into the saved report.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in ("png", "svg"):
        raise ValueError("请选择 PNG 或 SVG 文件格式")
    with _PLOT_LOCK:
        figure = build_evaluation_figure(report, view)
        FigureCanvasAgg(figure)
        try:
            if limits is not None:
                for axis, (xlim, ylim) in zip(figure.axes, limits):
                    axis.set_xlim(xlim)
                    axis.set_ylim(ylim)
            figure.savefig(path, format=suffix, dpi=180, facecolor=figure.get_facecolor())
        finally:
            figure.clear()
