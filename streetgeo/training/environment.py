"""Parameter groups and frozen-prefix behavior for environment specialist fine-tuning."""
from streetgeo.datasets.road_mask import RoadMask, MASK_FILL

LEGACY_RECIPE = {"version": 1, "mask": "oriented_source_bottom_third", "fill_rgb": list(MASK_FILL),
                 "update_order": ["full", "environment"]}


def recipe_for(mask):
    return {"version": 2, "mask": "oriented_source_bottom_trapezoid", "fill_rgb": list(MASK_FILL),
            "top_width": mask.top_width, "bottom_width": mask.bottom_width, "height": mask.height,
            "update_order": ["full", "environment"]}


def resolve_recipe(run, state=None, *, explicit_geometry=False):
    mask = RoadMask(run["road_mask_top_width"], run["road_mask_bottom_width"], run["road_mask_height"])
    recipe = recipe_for(mask)
    if state is not None:
        saved = state.get("environment_recipe", LEGACY_RECIPE)
        if saved == LEGACY_RECIPE:
            if explicit_geometry:
                raise ValueError("旧矩形断点按原策略续训；切换梯形请用 --init-from 新建微调阶段")
            return saved, mask, True
        if saved != recipe:
            raise ValueError("environment training recipe changed; start a new stage with --init-from")
    return recipe, mask, False


def parameter_groups(model, *, unfreeze_last, learning_rate, visual_learning_rate, weight_decay):
    if not 1 <= unfreeze_last <= len(model.encoder.visual_blocks):
        raise ValueError("unfreeze-last must be between 1 and the checkpoint's number of visual layers")
    start = len(model.encoder.visual_blocks) - unfreeze_last
    visual, downstream = [], []
    for name, parameter in model.named_parameters():
        terminal = name.startswith("encoder.visual_norm.") or any(
            name.startswith(f"encoder.visual_blocks.{number}.") for number in range(start, start + unfreeze_last))
        trainable = terminal or not name.startswith("encoder.") or name.startswith((
            "encoder.pool_attention.", "encoder.pool_queries", "encoder.heading_embedding.",
            "encoder.fusion_blocks.", "encoder.fusion_norm.", "encoder.geo_token"))
        parameter.requires_grad_(trainable)
        if trainable:
            (visual if terminal else downstream).append(parameter)
    return [{"params": visual, "lr": visual_learning_rate, "base_learning_rate": visual_learning_rate,
             "weight_decay": weight_decay, "name": "terminal_visual"},
            {"params": downstream, "lr": learning_rate, "base_learning_rate": learning_rate,
             "weight_decay": weight_decay, "name": "pool_fusion_location"}]


def train_mode(model, unfreeze_last):
    model.train()
    for block in model.encoder.visual_blocks[:-unfreeze_last]:
        block.eval()
