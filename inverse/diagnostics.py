from __future__ import annotations


def summarize_inverse_result(result, top_k: int = 5) -> dict:
    """Return a compact dictionary with best and top-k candidate diagnostics."""
    top_k = max(0, int(top_k))
    ordered = sorted(result.candidates, key=lambda candidate: candidate.residual_norm)
    return {
        "best_candidate_index": result.best_candidate_index,
        "residual_norm": result.residual_norm,
        "relative_residual": result.relative_residual,
        "estimated_position": result.estimated_position.tolist(),
        "estimated_moment": result.estimated_moment.tolist(),
        "top_candidates": [candidate.to_row() for candidate in ordered[:top_k]],
    }


def format_inverse_summary(result, top_k: int = 5) -> str:
    """Format inverse diagnostics as human-readable text."""
    summary = summarize_inverse_result(result, top_k=top_k)
    lines = [
        f"best candidate: {summary['best_candidate_index']}",
        f"estimated position: {summary['estimated_position']}",
        f"estimated moment: {summary['estimated_moment']}",
        f"residual norm: {summary['residual_norm']:.6g}",
        f"relative residual: {summary['relative_residual']:.6g}",
        f"top {len(summary['top_candidates'])} candidates by residual:",
    ]
    for row in summary["top_candidates"]:
        lines.append(
            "  "
            f"#{row['candidate_index']}: residual={row['residual_norm']:.6g}, "
            f"rel={row['relative_residual']:.6g}, "
            f"position=({row['x']:.6g}, {row['y']:.6g}, {row['z']:.6g})"
        )
    return "\n".join(lines)
