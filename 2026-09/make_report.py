# -*- coding: utf-8 -*-
"""노트북들이 남긴 결과 JSON을 모아 최종 비교표를 만든다.

표본 수가 하나라도 다르면 에러를 내고 멈춘다. 서로 다른 구간에서 잰 수치를
한 표에 올린 것이 원 논문의 핵심 결함이었으므로, 그 실수를 코드로 막는다.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import common


METRIC_KEYS = (
    "mae",
    "mse",
    "rmse",
    "mape",
    "mspe",
    "unit",
    "n_samples",
    "n_skipped",
)

NUMERIC_METRIC_KEYS = (
    "mae",
    "mse",
    "rmse",
    "mape",
    "mspe",
)

TABLE_COLUMNS = [
    "모델",
    "MAE(원)",
    "RMSE(원)",
    "MAPE(%)",
    "naive대비MAE배수",
    "판정",
]


def _as_float(value: Any, label: str) -> float:
    """유한한 실수 값을 반환한다."""
    if isinstance(value, bool):
        raise ValueError(f"{label}는 숫자여야 합니다.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}는 숫자여야 합니다.") from exc

    if not np.isfinite(number):
        raise ValueError(f"{label}는 유한한 숫자여야 합니다.")

    return number


def _as_int(value: Any, label: str) -> int:
    """정수 값을 반환한다."""
    number = _as_float(value, label)

    if not number.is_integer():
        raise ValueError(f"{label}는 정수여야 합니다.")

    return int(number)


def _require_metrics(metrics: Any, label: str) -> Mapping[str, Any]:
    """지표 dict 구조를 확인한다."""
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} 지표가 dict 형태가 아닙니다.")

    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"{label} 지표에 필수 키가 없습니다: {', '.join(missing)}")

    for key in NUMERIC_METRIC_KEYS:
        _as_float(metrics[key], f"{label}.{key}")

    _as_int(metrics["n_samples"], f"{label}.n_samples")
    _as_int(metrics["n_skipped"], f"{label}.n_skipped")

    if not isinstance(metrics["unit"], str):
        raise ValueError(f"{label}.unit은 문자열이어야 합니다.")

    return metrics


def _validate_result(result: Any, index: int) -> Mapping[str, Any]:
    """결과 dict 구조를 확인한다."""
    if not isinstance(result, Mapping):
        raise ValueError(f"{index}번째 결과가 dict 형태가 아닙니다.")

    required_keys = (
        "name",
        "model",
        "naive",
        "mae_ratio",
        "rmse_ratio",
        "mape_ratio",
        "beats_naive",
        "unit",
    )
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise ValueError(f"{index}번째 결과에 필수 키가 없습니다: {', '.join(missing)}")

    _require_metrics(result["model"], f"{index}번째 model")
    _require_metrics(result["naive"], f"{index}번째 naive")

    _as_float(result["mae_ratio"], f"{index}번째 mae_ratio")
    _as_float(result["rmse_ratio"], f"{index}번째 rmse_ratio")
    _as_float(result["mape_ratio"], f"{index}번째 mape_ratio")

    if not isinstance(result["beats_naive"], (bool, np.bool_)):
        raise ValueError(f"{index}번째 beats_naive는 bool이어야 합니다.")

    if not isinstance(result["unit"], str):
        raise ValueError(f"{index}번째 unit은 문자열이어야 합니다.")

    return result


# naive 기준선 비교 허용 오차.
# MinMax 노트북은 true가 float32 텐서를 거쳐 돌아오고 RevIN 노트북은 float64 그대로라
# 같은 기준선인데도 38,595원 중 0.045원(상대 1.2e-06) 차이가 난다. 정밀도 문제이지 오류가 아니다.
# 정렬이나 표본이 실제로 달랐다면 차이가 이 허용치보다 몇 자릿수 크게 난다.
NAIVE_RTOL = 1e-4


def _same_float(left: Any, right: Any) -> bool:
    """두 실수 값이 부동소수점 정밀도 범위 안에서 같은지 본다."""
    return bool(
        np.isclose(float(left), float(right), rtol=NAIVE_RTOL, atol=1e-6, equal_nan=True)
    )


def _validate_sample_counts(results: list) -> int:
    """모든 모델과 naive의 표본 수를 검증한다."""
    model_samples = [
        _as_int(result["model"]["n_samples"], "model.n_samples") for result in results
    ]
    naive_samples = [
        _as_int(result["naive"]["n_samples"], "naive.n_samples") for result in results
    ]

    if len(set(model_samples)) != 1:
        raise ValueError("모델별 n_samples가 다릅니다. 같은 표에 비교할 수 없습니다.")

    if len(set(naive_samples)) != 1:
        raise ValueError("naive별 n_samples가 다릅니다. 같은 표에 비교할 수 없습니다.")

    if model_samples[0] != naive_samples[0]:
        raise ValueError("모델과 naive의 n_samples가 다릅니다. 같은 표에 비교할 수 없습니다.")

    return model_samples[0]


def _validate_naive_metrics(results: list) -> Mapping[str, Any]:
    """모든 결과의 naive 지표 동일성을 검증한다."""
    reference = results[0]["naive"]

    for index, result in enumerate(results[1:], start=2):
        current = result["naive"]

        for key in NUMERIC_METRIC_KEYS:
            if not _same_float(reference[key], current[key]):
                raise ValueError(f"naive.{key} 값이 {index}번째 결과에서 다릅니다.")

        if reference["unit"] != current["unit"]:
            raise ValueError(f"naive.unit 값이 {index}번째 결과에서 다릅니다.")

        if int(reference["n_samples"]) != int(current["n_samples"]):
            raise ValueError(f"naive.n_samples 값이 {index}번째 결과에서 다릅니다.")

        if int(reference["n_skipped"]) != int(current["n_skipped"]):
            raise ValueError(f"naive.n_skipped 값이 {index}번째 결과에서 다릅니다.")

    return reference


def _validate_units(results: list) -> None:
    """표 머리글이 원 단위이므로 모든 지표가 원 단위인지 확인한다."""
    for index, result in enumerate(results, start=1):
        units = {
            "unit": result["unit"],
            "model.unit": result["model"]["unit"],
            "naive.unit": result["naive"]["unit"],
        }
        for label, unit in units.items():
            if unit != "원":
                raise ValueError(
                    f"{index}번째 결과의 {label} 값이 '원'이 아닙니다({unit}). "
                    "표 머리글이 원 단위라 같은 표에 올릴 수 없습니다."
                )


def _format_amount(value: Any) -> str:
    """금액을 천 단위 구분 형식으로 만든다."""
    return f"{_as_float(value, '금액'):,.2f}"


def _format_mape(value: Any) -> str:
    """MAPE를 소수 넷째 자리까지 표시한다."""
    return f"{_as_float(value, 'MAPE'):.4f}"


def _format_ratio(value: Any) -> str:
    """비교 배수를 소수 셋째 자리까지 표시한다."""
    return f"{_as_float(value, 'MAE 배수'):.3f}"


def _sorted_results(results: list) -> list:
    """MAE 배수가 낮은 순서로 결과를 정렬한다."""
    return sorted(results, key=lambda result: _as_float(result["mae_ratio"], "mae_ratio"))


def _build_table(sorted_results: list, naive: Mapping[str, Any]) -> pd.DataFrame:
    """최종 표시용 표를 만든다."""
    rows = [
        {
            "모델": "naive 기준선",
            "MAE(원)": _format_amount(naive["mae"]),
            "RMSE(원)": _format_amount(naive["rmse"]),
            "MAPE(%)": _format_mape(naive["mape"]),
            "naive대비MAE배수": "1.000",
            "판정": "기준선",
        }
    ]

    for result in sorted_results:
        model = result["model"]
        model_name = str(result["name"]).strip() or "이름 없음"

        rows.append(
            {
                "모델": model_name,
                "MAE(원)": _format_amount(model["mae"]),
                "RMSE(원)": _format_amount(model["rmse"]),
                "MAPE(%)": _format_mape(model["mape"]),
                "naive대비MAE배수": _format_ratio(result["mae_ratio"]),
                "판정": (
                    "기준선보다 나음"
                    if bool(result["beats_naive"])
                    else "기준선보다 나쁨"
                ),
            }
        )

    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _markdown_escape(value: Any) -> str:
    """마크다운 표 셀 문자열을 안전하게 만든다."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _to_markdown(table: pd.DataFrame) -> str:
    """외부 의존성 없이 마크다운 표를 만든다."""
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"

    rows = [header, separator]

    for _, row in table.iterrows():
        values = [_markdown_escape(row[column]) for column in table.columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows) + "\n"


def _load_results() -> list:
    """results 폴더의 결과 JSON을 읽는다."""
    try:
        collected = common.collect_results("results")
    except FileNotFoundError:
        return []

    if collected is None:
        return []

    if isinstance(collected, Mapping):
        return [collected]

    return list(collected)


def main() -> None:
    """결과를 검증하고 최종 표를 저장한다."""
    results = _load_results()

    if not results:
        print("결과 JSON이 없습니다. results 폴더를 확인하세요.")
        return

    validated_results = [
        _validate_result(result, index) for index, result in enumerate(results, start=1)
    ]

    n_samples = _validate_sample_counts(validated_results)
    naive = _validate_naive_metrics(validated_results)
    _validate_units(validated_results)
    sorted_results = _sorted_results(validated_results)

    table = _build_table(sorted_results, naive)

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "final_table.csv"
    markdown_path = output_dir / "final_table.md"

    print("테스트 표본 수: %d (전 모델 동일)" % n_samples)
    print()
    print(table.to_string(index=False))
    print()

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(_to_markdown(table), encoding="utf-8")

    win_count = sum(bool(result["beats_naive"]) for result in sorted_results)
    best_result = sorted_results[0]
    best_name = str(best_result["name"]).strip() or "이름 없음"
    best_ratio = _as_float(best_result["mae_ratio"], "최고 모델 MAE 배수")

    print(
        f"요약: {len(sorted_results)}개 모델 중 {win_count}개가 기준선을 이겼습니다. "
        f"최고 모델: {best_name}, naive대비 MAE 배수: {best_ratio:.3f}"
    )
    print(f"저장: {csv_path}, {markdown_path}")


if __name__ == "__main__":
    main()
