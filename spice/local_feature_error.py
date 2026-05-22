from __future__ import annotations


def softmax_delta_expr(target_expr: str, y_expr: str) -> str:
    return f"({target_expr})*(1-({y_expr})) - (1-({target_expr}))*{{SOFTMAX_NEGATIVE_SCALE}}*({y_expr})"


def mean_centered_expr(exprs: list[str], index: int) -> str:
    if not exprs:
        raise ValueError("exprs must be non-empty")
    summed = " + ".join(f"({expr})" for expr in exprs)
    return f"({exprs[index]}) - (({summed})/{len(exprs)})"
