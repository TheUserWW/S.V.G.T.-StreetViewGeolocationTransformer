"""Developer-only, batched evaluation of local OSV-5M official test images."""

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time

from streetgeo.inference.osv_test import scan_test_samples, TestScanCancelled, distance_km
from streetgeo.inference.evaluation_charts import distribution_data, save_evaluation_charts
from streetgeo.paths import ROOT

# Global-map approximation measured in PIGEON, appendix F.1. GeoGuessr map
# creators can change scoring radii, so this is a fixed, reproducible World
# profile rather than a claim to match every current/custom game map.
SCORE_SCALE_KM = 1492.7
SCORE_SOURCE = "https://arxiv.org/html/2307.05845v3#A6.SS1"
SCORE_RULES_SOURCE = "https://latb.io/geoguessr/articles/the-maths#score"
THRESHOLDS_KM = (1, 25, 200, 750, 2500)
RESULT_FIELDS = ("osv5m_id", "image", "country", "status", "true_latitude", "true_longitude",
                 "predicted_latitude", "predicted_longitude", "top1_error_km", "top1_score",
                 "best_of_5_error_km", "candidates_json", "error")


def geoguessr_score(error_km):
    """GeoGuessr World profile: integer points, 5000 maximum, 25 m minimum grace.

    Use positive half-up rounding (JavaScript Math.round), not Python's
    ties-to-even round. The decay scale is the published World approximation.
    """
    if not math.isfinite(error_km) or error_km < 0:
        raise ValueError("距离误差必须为非负有限数")
    if error_km <= .025:
        return 5000
    return max(0, min(5000, math.floor(5000 * math.exp(-error_km / SCORE_SCALE_KM) + .5)))


def describe(values):
    """Linear-interpolated quantiles and population standard deviation."""
    values = sorted(values)
    if not values:
        return {"count": 0, **dict.fromkeys(("mean", "median", "min", "max", "std", "rmse",
                                            "p25", "p75", "p90", "p95", "p99"))}

    def quantile(p):
        position = (len(values) - 1) * p
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": values[0], "max": values[-1], "std": statistics.pstdev(values),
            "rmse": math.sqrt(statistics.fmean(value * value for value in values)),
            **{f"p{p}": quantile(p / 100) for p in (25, 75, 90, 95, 99)}}


def summarize_errors(errors):
    thresholds = [sum(value <= threshold for value in errors) for threshold in THRESHOLDS_KM]
    bins = [thresholds[0], *(b - a for a, b in zip(thresholds, thresholds[1:])), len(errors) - thresholds[-1]]
    labels = ("0–1", "1–25", "25–200", "200–750", "750–2500", ">2500")
    scores = [geoguessr_score(value) for value in errors]
    return {"error_km": describe(errors), "score": describe(scores),
            "total_score": sum(scores), "maximum_total_score": 5000 * len(scores),
            "perfect_scores": {"count": scores.count(5000),
                               "fraction": scores.count(5000) / len(scores) if scores else None},
            "distance_bins_km": {label: count for label, count in zip(labels, bins)},
            "within_km": {str(threshold): {"count": sum(value <= threshold for value in errors),
                                           "fraction": (sum(value <= threshold for value in errors) / len(errors)
                                                        if errors else None)}
                          for threshold in THRESHOLDS_KM}}


def format_evaluation_report(report):
    status = {"completed": "遍历完成", "cancelled": "已停止（部分结果）", "failed": "评估失败（部分结果）"}
    coverage = report.get("coverage") or {}
    lines = ["Street-View Geolocation Transformer (S.V.G.T.) · 全量 test 评估",
             f"状态：{status.get(report['status'], report['status'])}",
             f"模型：{report['checkpoint']}", f"测试目录：{report['test_directory']}",
             f"设备：{report.get('device', '—')} · batch {report['batch_size']}",
             "预测范围：" + ("Google 官方街景覆盖国家／地区（离线粗筛）" if report.get("google_coverage_only") else "全球"),
             f"耗时：{report['elapsed_seconds']:.1f} 秒",
             f"成功 {report['successful']} · 图片失败 {report['failed']} · 未运行 "
             + (str(report['not_run']) if report['not_run'] is not None else "未知（扫描未完成）"),
             f"CSV 行数 {coverage.get('metadata_rows', '—')} · 可测试图片 {coverage.get('available_samples', '—')}",
             f"缺失图片 {coverage.get('missing_image', '—')} · 无效 ID/坐标 {coverage.get('invalid_id_or_coordinates', '—')}"
             f" · 重复 ID {coverage.get('duplicate_id', '—')}",
             "完整测试集评估：" + ("是" if report['full_test_complete'] else "否，请结合覆盖数量解释结果"), "",
             "以下统计以成功评估样本为分母；缺失、无效、失败和未运行样本不计分。",
             "Top-1 = 模型概率最高的候选。距离越小越好，评分越高越好。",
             "GeoGuessr 全球地图口径（近似复现）：round_half_up(5000 × exp(-距离误差 km / 1492.7))。",
             "每张图片计整数分，范围 0–5000；25 米以内满分，近距离也可能因四舍五入满分。",
             "1492.7 来自 PIGEON 对全球地图的实测近似；不同地图可自定义满分半径，不能保证与所有地图逐分一致。",
             f"评分参考：{SCORE_SOURCE}",
             f"距离与满分范围参考：{SCORE_RULES_SOURCE}",
             "分位数采用线性插值；标准差为总体标准差。", ""]
    if report.get("error"):
        lines.extend([f"错误：{report['error']}", ""])
    for title, key in (("Top-1 距离误差 / km", "error_km"), ("Top-1 评分 / 5000", "score")):
        stats = report["top1"][key]
        lines.append(title)
        for name, text in (("mean", "平均数"), ("median", "中位数"), ("min", "最小值"), ("max", "最大值"),
                           ("std", "标准差"), ("p25", "P25"), ("p75", "P75"), ("p90", "P90"),
                           ("p95", "P95"), ("p99", "P99")):
            lines.append(f"  {text}: {stats[name]:.3f}" if stats[name] is not None else f"  {text}: —")
        if key == "error_km" and stats["rmse"] is not None:
            lines.append(f"  RMSE: {stats['rmse']:.3f}")
        lines.append("")
    lines.append("距离阈值命中率（Top-1）")
    for threshold, item in report["top1"]["within_km"].items():
        ratio = f"{item['fraction']:.2%}" if item['fraction'] is not None else "—"
        lines.append(f"  ≤ {threshold} km: {item['count']} / {report['successful']} ({ratio})")
    perfect = report["top1"]["perfect_scores"]
    lines.extend(["", f"总得分：{report['top1']['total_score']} / {report['top1']['maximum_total_score']}",
                  f"满分（5000）：{perfect['count']} 张"])
    lines.extend(["", "距离误差分布（除第一段外，左开右闭）"])
    lines.extend(f"  {label} km: {count}" for label, count in report["top1"]["distance_bins_km"].items())
    oracle = report["best_of_5_error_km"]
    lines.extend(["", "前 5 候选最小距离（使用真实位置选出的理想上限，不能当作 Top-1 成绩）"])
    if oracle["count"]:
        lines.append(f"  平均 {oracle['mean']:.3f} km · 中位数 {oracle['median']:.3f} km")
    lines.extend(["", "按真实国家/地区分组（来自 test.csv；每组成功样本的 Top-1 统计）",
                  "  国家/地区 | 数量 | 平均误差 km | 中位误差 km | 平均分 | 中位分"])
    for country, group in sorted(report["by_country"].items()):
        error, score = group["error_km"], group["score"]
        lines.append(f"  {country} | {error['count']} | {error['mean']:.3f} | {error['median']:.3f}"
                     f" | {score['mean']:.2f} | {score['median']:.2f}")
    lines.extend(["", f"报告目录：{report['output_directory']}",
                  "summary.json：完整统计与评估口径；results.csv：逐图预测/评分/失败原因；",
                  "excluded.csv：元数据中缺失、无效和重复的行；report.txt：本报告。"])
    if report.get("charts"):
        lines.append("statistics_overview / statistics_countries：可视化统计图（PNG 与 SVG）。")
    if report.get("visualization_error"):
        lines.append(f"图表生成失败，数值报告已保留：{report['visualization_error']}")
    return "\n".join(lines)


def _save_report(report):
    folder = Path(report["output_directory"])
    for filename, content in (("summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)),
                              ("report.txt", format_evaluation_report(report))):
        temporary = folder / (filename + ".tmp")
        temporary.write_text(content + "\n", encoding="utf-8")
        os.replace(temporary, folder / filename)


def evaluate_test_set(checkpoint, directory, *, device="auto", batch_size=16, output_root=None,
                      progress=lambda event: None, cancelled=lambda: False, google_coverage_only=False):
    """Evaluate each eligible local test ID once, loading a fixed model once.

    Only image tensors enter model.predict; labels are used afterwards. Image
    decode failures are recorded and skipped. Model/device failures stop the run
    and preserve a clearly marked partial report. Completed batches flush to CSV.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("评估 batch 应为 1～128 的整数")
    path = Path(checkpoint).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    folder = Path(output_root) if output_root else ROOT / ".gui" / "evaluations"
    output = folder / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(path), "test_directory": str(Path(directory).expanduser().resolve()),
              "batch_size": batch_size, "requested_device": device, "output_directory": str(output.resolve()),
              "google_coverage_only": bool(google_coverage_only),
              "prediction_scope": "google_streetview_regions" if google_coverage_only else "world",
              "coverage": None, "successful": 0, "failed": 0, "not_run": 0,
              "score_profile": "geoguessr_world_approximation", "score_maximum": 5000,
              "score_formula": "5000 if d_km <= 0.025 else floor(5000 * exp(-d_km / 1492.7) + 0.5)",
              "score_scale_km": SCORE_SCALE_KM, "score_source": SCORE_SOURCE, "score_rules_source": SCORE_RULES_SOURCE,
              "score_rounding": "per_sample_half_up_to_integer",
              "score_note": "Fixed World profile based on published measurements; custom map scoring can differ.",
              "metric_denominator": "successful_samples_only", "quantile_method": "linear_(n-1)*p",
              "full_test_complete": False, "elapsed_seconds": 0.0}
    errors, oracle_errors, countries = [], [], {}
    samples = []
    with (output / "excluded.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerow(("csv_line", "osv5m_id", "reason"))

    def notify(phase):
        elapsed = time.perf_counter() - start
        done = report["successful"] + report["failed"]
        progress({"phase": phase, "processed": done, "total": len(samples),
                  "successful": report["successful"], "failed": report["failed"],
                  "elapsed_seconds": elapsed, "images_per_second": done / elapsed if elapsed else 0,
                  "output_directory": str(output.resolve())})

    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        try:
            notify("scanning")
            scan = scan_test_samples(directory, cancelled=cancelled)
            samples, report["coverage"] = scan["samples"], scan["coverage"]
            with (output / "excluded.csv").open("w", encoding="utf-8-sig", newline="") as excluded_stream:
                excluded = csv.DictWriter(excluded_stream, fieldnames=("csv_line", "osv5m_id", "reason"))
                excluded.writeheader()
                excluded.writerows(scan["excluded"])
            if cancelled():
                raise TestScanCancelled()
            if not samples:
                raise ValueError("没有可测试的本地图片，请解压 test 图片分片并保留 test.csv")
            notify("loading_model")
            import torch
            from streetgeo.core.checkpoints import load_model
            from streetgeo.datasets.images import load_image
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stamp = path.stat()
            report["checkpoint_file"] = {"mtime_ns": stamp.st_mtime_ns, "bytes": stamp.st_size}
            model = load_model(path, device)
            prediction_options = {}
            if google_coverage_only:
                from streetgeo.inference.coverage import google_coverage_mask, coverage_details
                prediction_options["coordinate_filter"] = google_coverage_mask
                report["prediction_domain"] = coverage_details()
            after_load = path.stat()
            if (stamp.st_mtime_ns, stamp.st_size) != (after_load.st_mtime_ns, after_load.st_size):
                raise ValueError("模型加载期间 checkpoint 已更新，请选择固定模型副本后重试")
            report["device"] = str(next(model.parameters()).device)
            report["model_config"] = asdict(model.config)
            for offset in range(0, len(samples), batch_size):
                if cancelled():
                    break
                batch, tensors = [], []
                for sample in samples[offset:offset + batch_size]:
                    if cancelled():
                        break
                    try:
                        tensor = load_image(sample["image"], model.config.image_size)
                    except (OSError, ValueError) as error:
                        writer.writerow({"osv5m_id": sample["osv5m_id"], "image": sample["image"],
                                         "country": sample["country"], "status": "failed",
                                         "true_latitude": sample["latitude"], "true_longitude": sample["longitude"],
                                         "error": str(error)})
                        report["failed"] += 1
                    else:
                        batch.append(sample)
                        tensors.append(tensor)
                if cancelled():
                    break
                if batch:
                    inputs = torch.stack(tensors).unsqueeze(1).to(next(model.parameters()).device)
                    predictions = model.predict(inputs, top_k=min(5, model.config.total_cells), **prediction_options)
                    if len(predictions) != len(batch):
                        raise ValueError("模型返回的预测数量与 batch 不一致")
                    for sample, candidates in zip(batch, predictions):
                        if not candidates or any(not (math.isfinite(p.latitude) and -90 <= p.latitude <= 90
                                                     and math.isfinite(p.longitude) and -180 <= p.longitude <= 180
                                                     and math.isfinite(p.probability)) for p in candidates):
                            raise ValueError("模型返回了无效的预测坐标或概率")
                        distances = [distance_km(sample["latitude"], sample["longitude"], p.latitude, p.longitude)
                                     for p in candidates]
                        error, first = distances[0], candidates[0]
                        writer.writerow({"osv5m_id": sample["osv5m_id"], "image": sample["image"],
                                         "country": sample["country"], "status": "ok",
                                         "true_latitude": sample["latitude"], "true_longitude": sample["longitude"],
                                         "predicted_latitude": first.latitude, "predicted_longitude": first.longitude,
                                         "top1_error_km": error, "top1_score": geoguessr_score(error),
                                         "best_of_5_error_km": min(distances),
                                         "candidates_json": json.dumps([
                                             {"latitude": p.latitude, "longitude": p.longitude, "error_km": d,
                                              "cell_probability": p.probability, "cell_id": p.cell_id}
                                             for p, d in zip(candidates, distances)], allow_nan=False)})
                        errors.append(error)
                        oracle_errors.append(min(distances))
                        countries.setdefault(sample["country"] or "(unknown)", []).append(error)
                        report["successful"] += 1
                    del inputs, tensors
                stream.flush()
                notify("evaluating")
            report["status"] = "cancelled" if cancelled() else "completed"
        except TestScanCancelled:
            report["status"] = "cancelled"
        except Exception as error:
            report["status"], report["error"] = "failed", str(error)
            if "out of memory" in str(error).lower():
                report["error"] += "\n显存/内存不足，请调小评估 Batch 后重新运行。"
    report["elapsed_seconds"] = time.perf_counter() - start
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["attempted"] = report["successful"] + report["failed"]
    report["not_run"] = len(samples) - report["attempted"] if report["coverage"] is not None else None
    report["top1"] = summarize_errors(errors)
    report["visualization"] = distribution_data(errors, [geoguessr_score(error) for error in errors])
    report["best_of_5_error_km"] = describe(oracle_errors)
    report["by_country"] = {country: summarize_errors(values) for country, values in countries.items()}
    coverage = report["coverage"]
    report["full_test_complete"] = bool(report["status"] == "completed" and coverage and not report["failed"]
                                        and not coverage["missing_image"] and not coverage["invalid_id_or_coordinates"])
    _save_report(report)
    try:
        notify("rendering_charts")
        report["charts"] = save_evaluation_charts(report)
    except Exception as error:
        # Plotting must not invalidate a completed evaluation or discard CSVs.
        report["visualization_error"] = str(error)
    _save_report(report)
    return report
