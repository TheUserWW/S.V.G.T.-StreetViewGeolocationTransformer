"""Dependency-free language strings for the portable client and launcher."""
import json
import locale
from pathlib import Path

LANGUAGES = {"zh": "中文", "en": "English"}
STRINGS = {
    "title": ("街景位置预测", "Street-view prediction"),
    "subtitle": ("选择街景，在地图上查看预测位置", "Choose a street view and find its predicted location"),
    "language": ("语言 / Language", "Language / 语言"),
    "images": ("街景图片", "Street-view images"),
    "model": ("模型文件", "Model file"),
    "choose_model": ("选择模型文件", "Choose model file"),
    "model_files": ("S.V.G.T. 模型文件", "S.V.G.T. model files"),
    "model_hint": ("客户端不附带模型，请选择单独下载的模型文件。", "Models are distributed separately. Select your downloaded model file."),
    "model_required": ("请先选择模型文件，再导入街景图片", "Choose a model file, then select street-view images"),
    "choose": ("选择街景图片", "Choose street-view images"),
    "image_hint": ("选择 1–4 张同一地点的街景图片。", "Choose 1–4 street views of the same location."),
    "placeholder": ("尚未选择图片", "No images selected"),
    "image_files": ("图片文件", "Image files"),
    "all_files": ("所有文件", "All files"),
    "selected": ("已选择 {count} 张图片", "{count} image(s) selected"),
    "predict": ("预测位置", "Predict location"),
    "coverage_only": ("仅官方街景覆盖国家／地区", "Google-covered countries/regions only"),
    "coverage_hint": ("按内置国家／地区名单离线筛选，含部分覆盖地区；不保证具体道路有街景。取消勾选可恢复全球预测。",
                      "Offline country/region allowlist, including partial coverage. Individual roads are not verified. Uncheck for worldwide predictions."),
    "coverage_empty": ("此模型没有落在所选覆盖国家／地区内的候选坐标。请关闭范围限制或更换模型后重试。",
                       "This model has no candidate coordinates in the selected countries/regions. Disable the restriction or choose another model."),
    "explain": ("关注区域 / 遮挡分析", "Attention / occlusion"),
    "explaining": ("正在分析关注区域…", "Analyzing attention and occlusion…"),
    "explanation_done": ("关注区域分析完成，请查看解释窗口", "Analysis complete. See the explanation window."),
    "explanation_cancelled": ("分析已取消，已有结果保留", "Analysis cancelled. Previous results are preserved."),
    "explanation_cancelling": ("正在取消分析，等待当前操作结束…", "Cancelling analysis. Waiting for the current operation…"),
    "explanation_failed": ("分析失败，请尝试重新分析当前输入，并检查图片、模型和保存路径。详情见 .runtime/prediction.log。", "Analysis failed. Try Analyze current inputs again and check images, model and save path. See .runtime/prediction.log for details."),
    "results": ("预测位置", "Predicted locations"),
    "results_hint": ("地图上的编号与下方候选一致。点击候选可放大查看。", "Map numbers match the candidates below. Select one to zoom in."),
    "no_results": ("预测结果将在地图上显示", "No predictions yet"),
    "world": ("全球总览", "World view"),
    "ready": ("请选择街景图片", "Choose street-view images to begin"),
    "loaded": ("图片已就绪，点击“预测位置”", "Images ready. Select Predict location."),
    "preparing": ("正在准备预测…", "Preparing prediction…"),
    "loading": ("正在加载所选模型，首次预测可能需要稍候…", "Loading the selected model. The first prediction may take a moment…"),
    "analyzing": ("正在分析街景…", "Analyzing street views…"),
    "done": ("预测完成 · 耗时 {seconds:.1f} 秒", "Prediction complete · {seconds:.1f} seconds"),
    "done_coverage": ("预测完成 · 已限制到官方街景覆盖国家／地区 · {seconds:.1f} 秒",
                      "Prediction complete · Google-covered countries/regions · {seconds:.1f} seconds"),
    "failed": ("预测失败，请检查图片和所选模型。", "Prediction failed. Please check the images and selected model."),
    "invalid_image": ("无法读取所选图片，请选择有效的 JPG、PNG 或 WebP 图片。", "Cannot read the selected image. Choose a valid JPG, PNG or WebP file."),
    "too_many": ("一次请选择 1–4 张同一地点的街景图片。", "Select 1–4 street views of the same location."),
    "missing_model": ("找不到所选模型，请选择有效的 S.V.G.T. 模型文件。", "The selected model is missing. Choose a valid S.V.G.T. model file."),
    "closing": ("正在等待本次预测结束后关闭…", "Waiting for the current prediction to finish before closing…"),
    "local": ("图片与预测均在本机处理", "Images stay on this computer"),
}

STRINGS.update({
    "daily_title": ("日常预测工作台", "Street-view workspace"),
    "daily_subtitle": ("粘贴截图，预测位置，点出真值。", "Paste a screenshot. Predict a location. Mark the truth."),
    "paste": ("粘贴截图  Ctrl+V", "Paste screenshot  Ctrl+V"),
    "import_hint": ("也可拖入图片。新截图开启新一局；补充视角请用下方按钮。", "Drop images here, too. A new screenshot starts a new round; add views with the button below."),
    "add_view": ("补充视角", "Add views"),
    "clear_images": ("清空 / 下一局", "Clear / next round"),
    "auto_predict": ("导入后自动预测", "Predict automatically on import"),
    "no_clipboard": ("剪贴板没有图片。先用 Win+Shift+S 截图，再按 Ctrl+V。", "No clipboard image. Capture with Win+Shift+S, then press Ctrl+V."),
    "clipboard_failed": ("暂时无法读取剪贴板，请稍后重试。", "Clipboard is unavailable. Please try again."),
    "working_wait": ("当前操作还在进行，请完成后再导入下一张。", "Wait for the current operation before importing another image."),
    "base_model": ("V4 基础模型", "V4 base model"),
    "personal_model": ("我的学习版", "My adapted model"),
    "choose_base": ("选择基础模型", "Choose base model"),
    "profile_hint": ("V4 原版保留；纠错只更新个人学习版。", "The base V4 stays intact. Corrections update your personal model."),
    "no_personal": ("还没有个人学习版。标注真实位置并学习后会自动建立。", "No personal model yet. Mark the true location and learn to create one."),
    "settings": ("设置", "Settings"),
    "history": ("最近记录", "Recent rounds"),
    "history_empty": ("还没有预测记录", "No recent rounds yet"),
    "history_hint": ("双击一条记录查看；图片和标点保存在本机，最多保留 100 局。", "Double-click a round to reopen it. Images and pins stay local; up to 100 rounds are retained."),
    "history_loaded": ("已打开历史记录 · 这是当时的预测，重新预测可使用当前模型。", "Opened a saved round. These are its original predictions; predict again to use the current model."),
    "history_missing": ("这条记录的原图片已移动或删除，无法重新打开。", "The original images were moved or deleted. This round cannot be reopened."),
    "history_save_failed": ("预测已完成，但本地记录保存失败，请检查磁盘空间。", "Prediction finished, but the round could not be saved. Check disk space."),
    "selected_result": ("首选坐标", "Top prediction"),
    "copy": ("复制坐标", "Copy coordinates"),
    "copied": ("已复制：{coordinates}", "Copied: {coordinates}"),
    "export": ("导出本局", "Export round"),
    "export_done": ("本局已导出", "Round exported"),
    "export_failed": ("保存失败，请检查文件路径和磁盘空间。", "Could not save. Check the path and disk space."),
    "focus": ("定位所选候选", "Focus selected guess"),
    "map_help": ("点击地图标点 · 滚轮缩放 · 拖动平移", "Click to place a pin · Scroll to zoom · Drag to pan"),
    "roads": ("道路底图", "Road map"),
    "manual_pin": ("手动标点", "Manual pin"),
    "truth_hint": ("游戏公布位置后，在地图上点出真值，或直接粘贴经纬度。", "After the reveal, mark the true location on the map or paste its coordinates."),
    "coordinates_hint": ("纬度, 经度", "latitude, longitude"),
    "apply_pin": ("应用坐标", "Place pin"),
    "clear_pin": ("清除标点", "Clear pin"),
    "pin_ready": ("已标点：{coordinates} · 点击“确认真值并学习”提交纠正。", "Pin placed: {coordinates}. Select Learn this correction to submit the truth."),
    "invalid_coordinates": ("请输入有效的“纬度, 经度”，例如 37.7749, -122.4194。", "Enter valid latitude, longitude, for example 37.7749, -122.4194."),
    "error_distance": ("候选 {rank} 距标点 {distance:,.1f} km", "Guess {rank} is {distance:,.1f} km from your pin"),
    "grid_probability": ("候选网格概率 {probability:.1%} · 非正确率", "Grid probability {probability:.1%} · not calibrated accuracy"),
    "learn": ("确认真值并学习", "Learn this correction"),
    "auto_learn": ("标点后自动学习", "Learn when I place a pin"),
    "auto_learn_hint": ("开启后，每次点击地图都会提交当前图片的真值。", "When enabled, each map click submits the current image's true location."),
    "learning": ("正在更新个人模型…", "Updating your personal model…"),
    "learning_done": ("第 {updates} 次纠错已保存 · 下次预测使用学习版。当前显示学习前结果。", "Correction {updates} saved. Next prediction uses your personal model. Current results are from before learning."),
    "learning_stats": ("已纠错 {updates} 次 · 记忆 {memory} 个地点", "{updates} corrections · {memory} remembered locations"),
    "learning_hint": ("仅使用游戏公布的真实位置。误点可撤销；学习效果需要新地点验证。", "Use the game's revealed location. Undo a mistaken click; verify improvement on new places."),
    "learning_failed": ("本次学习未保存。请检查模型与图片，详情见 prediction.log。", "This update was not saved. Check the model and images; see prediction.log."),
    "need_prediction": ("请先用当前模型预测这张图片，再标注真值学习。", "Predict this image with the current model before learning its true location."),
    "need_pin": ("请先在地图上标注真实位置。", "Mark the true location on the map first."),
    "already_learned": ("本局已学习。重新预测后可再次纠正。", "This round has been learned. Predict again before another correction."),
    "official_test": ("官方 test 只用于评估，不能加入个人学习。", "Official test images are for evaluation and cannot be used for personal learning."),
    "undo": ("撤销上次学习", "Undo last update"),
    "undone": ("已撤销学习 · 请重新预测。", "Update undone. Predict again."),
    "no_update": ("当前没有可撤销的个人学习记录。", "No personal update is available to undo."),
    "device": ("运算设备（auto 优先使用 GPU）", "Device (auto prefers GPU)"),
    "storage_hint": ("剪贴板截图、最近记录与个人模型仅保存在本机。分发包不含模型。", "Clipboard captures, recent rounds and personal models stay local. Distribution archives contain no models."),
    "preview": ("查看原图", "View original image"),
    "step_images": ("01  导入街景", "01  Add street views"),
    "step_predict": ("02  预测位置", "02  Predict"),
    "step_truth": ("03  标点与纠正", "03  Mark and correct"),
    "history_image": ("图片", "Image"),
    "history_time": ("时间", "Time"),
    "history_status": ("状态", "Status"),
    "history_predicted": ("已预测", "Predicted"),
    "history_marked": ("已标点", "Pinned"),
    "history_learned": ("已学习", "Learned"),
})

MAP_ENGLISH = {
    "真": "T", "真实位置": "True location", "手动标点": "Manual pin",
    "© OpenStreetMap contributors · 在线道路底图使用 OSM 的边界表达": "© OpenStreetMap contributors · OSM boundary depiction",
    "边界：UN Geospatial / OCHA · 离线概览": "Boundaries: UN Geospatial / OCHA · Offline map",
    "来源地图边界不代表联合国对领土归属的认可。": "Map boundaries do not imply UN endorsement of territorial status.",
    "国际边界": "Boundary", "虚线边界": "Dashed",
    "部分道路底图暂不可用，已保留离线轮廓": "Some road tiles are unavailable; offline boundaries remain visible.",
    "离线轮廓不含道路；可启用在线道路底图": "Offline map: boundaries only; roads are not included.",
}

LAUNCHER_ENGLISH = {
    "客户端运行目录指向了程序文件夹之外，请检查 .runtime": "The runtime directory points outside the client folder. Check .runtime.",
    "请使用 64 位 Python 3.10–3.15，推荐 Python 3.12。": "Use 64-bit Python 3.10–3.15. Python 3.12 is recommended.",
    "当前是 32 位 Python，请改用 64 位 Python 3.12。": "This Python is 32-bit. Use 64-bit Python 3.12.",
    "该客户端使用 Windows x64 依赖，请使用 x64 版本的 Python。": "This client requires Windows x64 and x64 Python.",
    "Python 缺少 tkinter；请在 Python 安装器中添加 Tcl/Tk and IDLE 组件。": "Python is missing tkinter. Add Tcl/Tk and IDLE using the Python installer.",
    "客户端已经启动或正在安装依赖，请查看已打开的窗口。": "The client is already running or installing dependencies. Check the open window.",
    "依赖检查无法运行：": "Dependency checks could not run: ",
    "[1/3] 正在创建客户端独立 Python 环境…": "[1/3] Creating the client's Python environment…",
    "[1/3] 已找到客户端独立 Python 环境。": "[1/3] Found the client's Python environment.",
    "[2/3] 正在安装或修复依赖，首次运行需要联网，请稍候…": "[2/3] Installing or repairing dependencies. Internet is required for first-time setup; please wait…",
    "[2/3] 依赖检查通过，无需重新安装。": "[2/3] Dependencies are ready; no installation needed.",
    "依赖仍无法加载：": "Dependencies still cannot be loaded: ",
    "。若提示 DLL load failed，请检查 Microsoft Visual C++ 运行库。": ". For DLL load failures, check the Microsoft Visual C++ runtime.",
    "缺少 best.pt，请把模型权重放回客户端文件夹。": "best.pt is missing. Restore the model to the client folder.",
    "客户端文件不完整，请重新解压整个文件夹。": "Client files are incomplete. Extract the entire client folder again.",
    "[3/3] 环境准备完成。": "[3/3] Environment ready.",
    "[3/3] 正在启动本地预测客户端…": "[3/3] Starting the prediction client…",
    "\n启动失败：": "\nStartup failed: ",
    "请把 .runtime/startup.log 发给开发者排查。": "Please provide .runtime/startup.log to the developer for troubleshooting.",
    "命令执行失败（{code}），详情见 {path}": "Command failed ({code}). See {path} for details.",
}


def read_language(settings_path):
    try:
        saved = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        if isinstance(saved, dict) and saved.get("language") in LANGUAGES:
            return saved["language"]
    except (OSError, ValueError, TypeError):
        pass
    try:
        system = locale.getlocale()[0] or ""
    except (ValueError, TypeError):
        system = ""
    return "zh" if system.lower().startswith("zh") else "en"


def text(key, language="zh", **values):
    return STRINGS[key][1 if language == "en" else 0].format(**values)


def map_text(source, language):
    if language != "en" and source == "离线轮廓不含道路；可启用在线道路底图":
        return "离线地图仅显示地理边界，不含道路。"
    return MAP_ENGLISH.get(source, source) if language == "en" else source
