def cohort_recommendations(effects: dict, th: float = 0.05) -> str:
    lines = ["# Kohorten‑Empfehlungen\n"]
    pvals = effects.get("_p", {})
    for k, v in effects.items():
        if k.startswith("_"):  # _n, _r2, _se, _p überspringen
            continue
        if abs(v) < th:
            continue
        star = ""
        if k in pvals and pvals[k] < 0.05:
            star = " **(signifikant)**"
        direction = "erhöhen" if v > 0 else "senken"
        lines.append(f"- **{k}** {direction} (β={v:+.2f}){star}")
    return "\n".join(lines)
