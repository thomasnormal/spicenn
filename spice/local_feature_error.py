from __future__ import annotations


def softmax_delta_expr(target_expr: str, y_expr: str) -> str:
    return f"({target_expr})*(1-({y_expr})) - (1-({target_expr}))*{{SOFTMAX_NEGATIVE_SCALE}}*({y_expr})"


def softmax_exp_expr(score_expr: str) -> str:
    return f"exp(({score_expr})/{{SOFTMAX_TEMPERATURE}})"


def max_expr(exprs: list[str]) -> str:
    if not exprs:
        raise ValueError("exprs must be non-empty")
    current = exprs[0]
    for expr in exprs[1:]:
        current = f"0.5*({current}+{expr}+abs({current}-{expr}))"
    return current


def clipped_unit_expr(expr: str) -> str:
    return f"(0.5*(({expr})+abs({expr}))-0.5*((({expr})-1)+abs(({expr})-1)))"


def target_margin_gate_expr(score_exprs: list[str], target_exprs: list[str]) -> str:
    if len(score_exprs) != len(target_exprs):
        raise ValueError("score and target expression counts must match")
    if len(score_exprs) < 2:
        raise ValueError("margin gate requires at least two classes")
    target_score = " + ".join(f"{target_expr}*{score_expr}" for target_expr, score_expr in zip(target_exprs, score_exprs))
    target_competitor = " + ".join(
        f"{target_exprs[index]}*({max_expr([expr for j, expr in enumerate(score_exprs) if j != index])})"
        for index in range(len(score_exprs))
    )
    deficit = f"({{SOFTMAX_MARGIN}}-(({target_score})-({target_competitor})))/({{SOFTMAX_MARGIN}}+1e-12)"
    return clipped_unit_expr(deficit)


def append_target_margin_gate(
    lines: list[str],
    gate_node: str,
    score_exprs: list[str],
    target_exprs: list[str],
) -> None:
    if len(score_exprs) != len(target_exprs):
        raise ValueError("score and target expression counts must match")
    if len(score_exprs) < 2:
        raise ValueError("margin gate requires at least two classes")
    target_score = " + ".join(f"{target_expr}*{score_expr}" for target_expr, score_expr in zip(target_exprs, score_exprs))
    competitor_terms = []
    for index, target_expr in enumerate(target_exprs):
        competitors = [expr for j, expr in enumerate(score_exprs) if j != index]
        current = competitors[0]
        for fold_index, competitor in enumerate(competitors[1:]):
            max_node = f"{gate_node}cmp{index}_{fold_index}"
            lines.append(f"B{max_node} {max_node} 0 V = {max_expr([current, competitor])}")
            current = f"V({max_node})"
        competitor_terms.append(f"{target_expr}*({current})")
    target_competitor = " + ".join(competitor_terms)
    deficit = f"({{SOFTMAX_MARGIN}}-(({target_score})-({target_competitor})))/({{SOFTMAX_MARGIN}}+1e-12)"
    lines.append(f"B{gate_node} {gate_node} 0 V = {clipped_unit_expr(deficit)}")


def mean_centered_expr(exprs: list[str], index: int) -> str:
    if not exprs:
        raise ValueError("exprs must be non-empty")
    summed = " + ".join(f"({expr})" for expr in exprs)
    return f"({exprs[index]}) - (({summed})/{len(exprs)})"


def class_centered_expr(exprs: list[str], index: int, mode: str) -> str:
    if mode == "none":
        return exprs[index]
    if mode == "mean":
        return f"({mean_centered_expr(exprs, index)})"
    raise ValueError("class centering mode must be 'none' or 'mean'")
