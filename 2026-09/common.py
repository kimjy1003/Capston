# -*- coding: utf-8 -*-
"""노트북 공통 파이프라인: 로드 / 분할 / 정규화 / 시퀀스 / 평가.

원 논문 노트북들이 제각각 하던 전처리와 평가를 하나로 모은 모듈이다.
핵심 원칙 2가지:
  1. 스케일러는 학습 구간에만 fit한다. 테스트 구간에는 transform만 한다.
  2. 모든 모델은 naive persistence 기준선과 같은 표본, 같은 단위로 비교한다.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _feature_columns() -> List[str]:
    """입력 피처 컬럼 순서를 반환한다."""
    return ["open", "high", "low", "close", "volume", "value", "ma_3", "ma_5"]


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """필수 컬럼 존재를 확인한다."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("필수 컬럼이 없습니다: {}".format(", ".join(missing)))


def _as_column(values: Any, label: str) -> np.ndarray:
    """평가 값을 열 벡터로 맞춘다."""
    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2 and array.shape[1] == 1:
        return array

    raise ValueError("{}은 (n, 1) 형태여야 합니다.".format(label))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """0 기준선에도 안전하게 비율을 계산한다."""
    if np.isnan(numerator) or np.isnan(denominator):
        return float("nan")

    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")

    return float(numerator / denominator)


def load_data(
    path: Union[str, Path] = "data/BTC_3min_2024-10.csv",
    ma: Tuple[int, int] = (3, 5),
) -> pd.DataFrame:
    """BTC 3분봉에 입력 피처와 middle 타깃을 추가한다."""
    if len(ma) != 2:
        raise ValueError("ma는 서로 다른 양의 정수 2개여야 합니다.")

    if any(
        isinstance(period, bool)
        or not isinstance(period, (int, np.integer))
        or period <= 0
        for period in ma
    ):
        raise ValueError("ma는 서로 다른 양의 정수 2개여야 합니다.")

    periods = (int(ma[0]), int(ma[1]))
    if periods[0] == periods[1]:
        raise ValueError("ma는 서로 다른 양의 정수 2개여야 합니다.")

    csv_path = Path(path)
    raw = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=[0],
        encoding="utf-8",
    )

    base_columns = ["open", "high", "low", "close", "volume", "value"]
    _require_columns(raw, base_columns)

    data = raw.loc[:, base_columns].apply(pd.to_numeric, errors="raise").copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = None

    for period in periods:
        data["ma_{}".format(period)] = data["close"].rolling(window=period).mean()

    data["middle"] = (data["high"] + data["low"]) / 2.0

    column_order = (
        base_columns
        + ["ma_{}".format(period) for period in periods]
        + ["middle"]
    )
    return data.loc[:, column_order].dropna().copy()


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """시계열 순서를 유지해 학습과 테스트 구간을 나눈다."""
    if len(df) < 2:
        raise ValueError("분할하려면 데이터가 2행 이상 필요합니다.")

    ratio = float(train_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("train_ratio는 0과 1 사이여야 합니다.")

    split_idx = int(len(df) * ratio)
    if split_idx == 0 or split_idx == len(df):
        raise ValueError("현재 데이터 길이에서는 학습과 테스트 구간이 모두 필요합니다.")

    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def fit_scalers(
    train_df: pd.DataFrame,
    kind: str = "minmax",
) -> Tuple[
    Union[MinMaxScaler, StandardScaler],
    Union[MinMaxScaler, StandardScaler],
]:
    """학습 구간으로만 입력과 타깃 스케일러를 적합한다."""
    if train_df.empty:
        raise ValueError("학습 데이터가 비어 있습니다.")

    feature_columns = _feature_columns()
    _require_columns(train_df, feature_columns + ["middle"])

    if not isinstance(kind, str):
        raise ValueError("kind는 minmax 또는 standard여야 합니다.")

    normalized_kind = kind.lower()
    if normalized_kind == "minmax":
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    elif normalized_kind == "standard":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    else:
        raise ValueError("kind는 minmax 또는 standard여야 합니다.")

    x_train = train_df.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    y_train = train_df.loc[:, ["middle"]].to_numpy(dtype=float, copy=True)

    # 학습 구간만 fit하여 테스트 정보 유입을 막는다.
    scaler_x.fit(x_train)
    scaler_y.fit(y_train)

    return scaler_x, scaler_y


def apply_scalers(
    df: pd.DataFrame,
    scaler_x: Union[MinMaxScaler, StandardScaler],
    scaler_y: Union[MinMaxScaler, StandardScaler],
) -> np.ndarray:
    """적합된 스케일러로 데이터에 transform만 적용한다."""
    feature_columns = _feature_columns()
    _require_columns(df, feature_columns + ["middle"])

    if df.empty:
        return np.empty((0, 9), dtype=float)

    x_values = df.loc[:, feature_columns].to_numpy(dtype=float, copy=True)
    y_values = df.loc[:, ["middle"]].to_numpy(dtype=float, copy=True)

    scaled_x = np.asarray(scaler_x.transform(x_values), dtype=float)
    scaled_y = np.asarray(scaler_y.transform(y_values), dtype=float)

    return np.concatenate((scaled_x, scaled_y), axis=1)


def make_sequences(
    arr: np.ndarray,
    seq_length: int,
    n_features: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """입력 윈도와 미래 타깃 시퀀스를 만든다."""
    data = np.asarray(arr)

    if data.ndim != 2:
        raise ValueError("arr은 (n, 열수) 형태의 2차원 배열이어야 합니다.")

    if (
        isinstance(seq_length, bool)
        or not isinstance(seq_length, (int, np.integer))
        or seq_length <= 0
    ):
        raise ValueError("seq_length는 양의 정수여야 합니다.")

    if (
        isinstance(n_features, bool)
        or not isinstance(n_features, (int, np.integer))
        or n_features <= 0
    ):
        raise ValueError("n_features는 양의 정수여야 합니다.")

    window = int(seq_length)
    feature_count = int(n_features)

    if data.shape[1] <= feature_count:
        raise ValueError("arr에는 입력 피처와 타깃 열이 모두 필요합니다.")

    n_sequences = data.shape[0] - window
    if n_sequences <= 0:
        return (
            np.empty((0, window, feature_count), dtype=data.dtype),
            np.empty((0, 1), dtype=data.dtype),
        )

    X = np.empty((n_sequences, window, feature_count), dtype=data.dtype)
    y = np.empty((n_sequences, 1), dtype=data.dtype)

    for i in range(n_sequences):
        X[i] = data[i : i + window, :feature_count]
        # 타깃은 입력 윈도 다음 시점이어야 동시점 정보 누수를 막는다.
        y[i] = data[i + window, [-1]]

    return X, y


def make_revin_sequences(
    df: pd.DataFrame,
    seq_length: int,
    n_features: int = 8,
    target_col: str = "middle",
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RevIN seq-to-one용 원 단위 시퀀스와 타깃 통계를 만든다.

    RevIN은 인스턴스(윈도)마다 채널별로 정규화하는 기법이라 sklearn 스케일러를 쓰지 않는다.
    입력 X는 원 단위로 내보내고 모델 안 RevIN이 정규화한다.
    타깃 middle은 입력 8채널에 없으므로, 그 윈도의 middle 평균/표준편차로 정규화하고
    같은 통계로 되돌린다. 통계는 입력 구간에서만 계산하므로 미래 정보가 섞이지 않는다.
    """
    if (
        isinstance(seq_length, bool)
        or not isinstance(seq_length, (int, np.integer))
        or seq_length <= 0
    ):
        raise ValueError("seq_length는 양의 정수여야 합니다.")

    if (
        isinstance(n_features, bool)
        or not isinstance(n_features, (int, np.integer))
        or n_features <= 0
    ):
        raise ValueError("n_features는 양의 정수여야 합니다.")

    eps_value = float(eps)
    if not np.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("eps는 양의 유한값이어야 합니다.")

    feature_count = int(n_features)
    window = int(seq_length)

    if df.shape[1] < feature_count:
        raise ValueError("입력 피처 수가 n_features보다 적습니다.")

    if target_col not in df.columns:
        raise ValueError("target_col이 데이터프레임에 없습니다: {}".format(target_col))

    input_columns = list(df.columns[:feature_count])
    if target_col in input_columns:
        raise ValueError("target_col은 입력 피처에 포함될 수 없습니다.")

    feature_values = df.iloc[:, :feature_count].to_numpy(dtype=float, copy=True)
    target_values = np.asarray(
        df[target_col].to_numpy(dtype=float, copy=True), dtype=float
    )

    if target_values.ndim != 1:
        raise ValueError("target_col은 중복되지 않은 단일 컬럼이어야 합니다.")

    n_sequences = feature_values.shape[0] - window
    if n_sequences <= 0:
        raise ValueError("seq_length보다 행 수가 커야 합니다.")

    X_raw = np.empty((n_sequences, window, feature_count), dtype=float)
    y_raw = np.empty((n_sequences, 1), dtype=float)
    win_mean = np.empty((n_sequences, 1), dtype=float)
    win_std = np.empty((n_sequences, 1), dtype=float)

    for i in range(n_sequences):
        target_window = target_values[i : i + window]

        X_raw[i] = feature_values[i : i + window]
        # 정렬 규칙은 make_sequences와 동일하다: y[i]는 i+seq_length 시점.
        y_raw[i, 0] = target_values[i + window]
        win_mean[i, 0] = np.mean(target_window)
        # RevIN과 같이 모표준편차(ddof=0)를 쓴다.
        win_std[i, 0] = np.std(target_window) + eps_value

    # make_sequences와 표본 수, 타깃 시점이 일치하는지 검증한다.
    reference_arr = np.column_stack((feature_values, target_values))
    _, reference_y = make_sequences(
        reference_arr, seq_length=window, n_features=feature_count
    )

    if y_raw.shape != reference_y.shape:
        raise ValueError("make_sequences와 RevIN 시퀀스의 표본 수가 다릅니다.")

    if not np.array_equal(y_raw, reference_y):
        raise ValueError("make_sequences와 RevIN 타깃 시점 정렬이 다릅니다.")

    return X_raw, y_raw, win_mean, win_std


def naive_baseline(
    X: np.ndarray,
    target_feature_idx: Optional[int] = None,
) -> np.ndarray:
    """스케일된 배열에서 단일 피처를 그대로 뽑는 용도. 서로 다른 스케일러를 거친 피처를 조합하지 말 것."""
    if target_feature_idx is None:
        raise ValueError(
            "target_feature_idx를 지정해야 합니다. 원 단위 middle 기준선은 "
            "naive_baseline_from_df를 사용하세요."
        )

    data = np.asarray(X)

    if data.ndim != 3:
        raise ValueError("X는 (n, seq_length, n_features) 형태여야 합니다.")

    if data.shape[1] == 0:
        raise ValueError("X의 seq_length는 1 이상이어야 합니다.")

    if (
        isinstance(target_feature_idx, bool)
        or not isinstance(target_feature_idx, (int, np.integer))
    ):
        raise ValueError("target_feature_idx는 정수여야 합니다.")

    feature_idx = int(target_feature_idx)
    if not -data.shape[2] <= feature_idx < data.shape[2]:
        raise ValueError("target_feature_idx가 입력 피처 범위를 벗어났습니다.")

    return data[:, -1, [feature_idx]].copy()


def naive_baseline_from_df(
    df: pd.DataFrame,
    seq_length: int,
    target_col: str = "middle",
) -> np.ndarray:
    """원 단위 persistence 기준선을 만든다.

    정렬 규칙: y[i]는 i+seq_length 시점, 기준선은 i+seq_length-1 시점.
    """
    if (
        isinstance(seq_length, bool)
        or not isinstance(seq_length, (int, np.integer))
        or seq_length <= 0
    ):
        raise ValueError("seq_length는 양의 정수여야 합니다.")

    if target_col not in df.columns:
        raise ValueError("target_col이 데이터프레임에 없습니다: {}".format(target_col))

    target_values = df[target_col].to_numpy()
    if target_values.ndim != 1:
        raise ValueError("target_col은 중복되지 않은 단일 컬럼이어야 합니다.")

    window = int(seq_length)
    n_rows = target_values.shape[0]

    if window > n_rows:
        raise ValueError("seq_length는 데이터 행 수보다 클 수 없습니다.")

    expected_y_count = n_rows - window
    baseline_values = target_values[window - 1 : -1]
    future_target_values = target_values[window:]

    if (
        baseline_values.shape[0] != expected_y_count
        or future_target_values.shape[0] != expected_y_count
    ):
        raise ValueError(
            "기준선과 make_sequences의 미래 타깃 행 수가 맞지 않습니다."
        )

    return baseline_values.reshape(-1, 1).copy()


def evaluate(
    pred: np.ndarray,
    true: np.ndarray,
    inverse_fn: Optional[Callable[[np.ndarray], Any]] = None,
    name: str = "",
    unit: Optional[str] = None,
) -> Dict[str, Any]:
    """예측값의 회귀 지표를 계산한다.

    unit을 생략하면 inverse_fn 유무로 추정한다(있으면 "원", 없으면 "정규화").
    이미 원 단위로 되돌린 값을 넘길 때는 unit="원"을 명시해야 라벨이 맞는다.
    """
    pred_values = _as_column(pred, "pred")
    true_values = _as_column(true, "true")

    if pred_values.shape != true_values.shape:
        raise ValueError("pred와 true의 형태가 같아야 합니다.")

    if pred_values.size == 0:
        raise ValueError("평가할 표본이 없습니다.")

    if inverse_fn is not None:
        pred_values = _as_column(inverse_fn(pred_values), "inverse pred")
        true_values = _as_column(inverse_fn(true_values), "inverse true")

        if pred_values.shape != true_values.shape:
            raise ValueError("역정규화 뒤 pred와 true의 형태가 같아야 합니다.")

    if unit is None:
        unit = "원" if inverse_fn is not None else "정규화"

    errors = pred_values - true_values
    absolute_errors = np.abs(errors)

    mae = float(np.mean(absolute_errors))
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))

    zero_tolerance = 1e-8
    valid_for_percentage = (
        np.isfinite(pred_values)
        & np.isfinite(true_values)
        & (np.abs(true_values) > zero_tolerance)
    )
    n_skipped = int((~valid_for_percentage).sum())

    if np.any(valid_for_percentage):
        percentage_errors = (
            errors[valid_for_percentage] / true_values[valid_for_percentage]
        )
        mape = float(np.mean(np.abs(percentage_errors)) * 100.0)
        mspe = float(np.mean(np.square(percentage_errors)) * 100.0)
    else:
        mape = float("nan")
        mspe = float("nan")

    return {
        "name": name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "unit": unit,
        "n_samples": int(true_values.shape[0]),
        "n_skipped": n_skipped,
    }


def compare_to_naive(
    model_pred: np.ndarray,
    naive_pred: np.ndarray,
    true: np.ndarray,
    inverse_fn: Optional[Callable[[np.ndarray], Any]] = None,
    name: str = "",
    unit: Optional[str] = None,
) -> Dict[str, Any]:
    """모델과 naive 기준선을 같은 단위에서 비교한다."""
    model_metrics = evaluate(
        model_pred,
        true,
        inverse_fn=inverse_fn,
        name=name,
        unit=unit,
    )
    naive_metrics = evaluate(
        naive_pred,
        true,
        inverse_fn=inverse_fn,
        name="naive",
        unit=unit,
    )

    mae_ratio = _safe_ratio(model_metrics["mae"], naive_metrics["mae"])
    rmse_ratio = _safe_ratio(model_metrics["rmse"], naive_metrics["rmse"])
    mape_ratio = _safe_ratio(model_metrics["mape"], naive_metrics["mape"])

    return {
        "name": name,
        "model": model_metrics,
        "naive": naive_metrics,
        "mae_ratio": mae_ratio,
        "rmse_ratio": rmse_ratio,
        "mape_ratio": mape_ratio,
        "beats_naive": bool(np.isfinite(mae_ratio) and mae_ratio < 1.0),
        "unit": model_metrics["unit"],
    }


def report_table(results: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """비교 결과를 표로 만들고 원본 수치는 attrs에 보관한다."""
    # 주의: beats_naive는 "모델이 naive를 이겼는가"이다. 컬럼명을 뒤집어 달지 말 것.
    columns = [
        "모델",
        "MAE",
        "RMSE",
        "MAPE(%)",
        "naive대비MAE배수",
        "모델이naive를이김",
    ]

    rows: List[Dict[str, Any]] = []

    for result in results:
        model_metrics = result.get("model")
        naive_metrics = result.get("naive")

        if not isinstance(model_metrics, Mapping) or not isinstance(
            naive_metrics, Mapping
        ):
            raise ValueError("compare_to_naive 결과 dict가 필요합니다.")

        model_name = result.get("name")
        if not model_name:
            model_name = model_metrics.get("name", "")

        rows.append(
            {
                "모델": str(model_name),
                "MAE": float(model_metrics["mae"]),
                "RMSE": float(model_metrics["rmse"]),
                "MAPE(%)": float(model_metrics["mape"]),
                "naive대비MAE배수": float(result["mae_ratio"]),
                "모델이naive를이김": bool(result["beats_naive"]),
            }
        )

    raw_table = pd.DataFrame(rows, columns=columns)
    table = raw_table.copy()

    numeric_columns = ["MAE", "RMSE", "MAPE(%)", "naive대비MAE배수"]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce").round(4)

    table.attrs["raw_values"] = raw_table.to_dict(orient="records")
    return table


def save_result(
    result: Mapping[str, Any],
    results_dir: Union[str, Path] = "results",
) -> Path:
    """compare_to_naive 결과 1건을 JSON으로 남긴다.

    노트북마다 따로 돌려도 나중에 collect_results로 한 표에 모을 수 있게 한다.
    파일명은 모델 이름에서 만들되 경로 구분자와 공백은 밑줄로 바꾼다.
    """
    name = str(result.get("name", "")).strip()
    if not name:
        raise ValueError("result에 name이 있어야 파일명을 만들 수 있습니다.")

    safe_name = name
    for bad in (" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        safe_name = safe_name.replace(bad, "_")

    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / "{}.json".format(safe_name)

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    return out_path


def collect_results(
    results_dir: Union[str, Path] = "results",
) -> List[Dict[str, Any]]:
    """results 폴더의 결과 JSON을 모두 읽어 리스트로 돌려준다."""
    directory = Path(results_dir)
    if not directory.is_dir():
        return []

    collected: List[Dict[str, Any]] = []
    for json_path in sorted(directory.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            collected.append(json.load(handle))

    return collected


if __name__ == "__main__":
    sequence_length = 288

    data = load_data()
    train_df, test_df = split_data(data)

    scaler_x, scaler_y = fit_scalers(train_df, kind="minmax")

    train_arr = apply_scalers(train_df, scaler_x, scaler_y)
    test_arr = apply_scalers(test_df, scaler_x, scaler_y)

    _, y_train = make_sequences(train_arr, seq_length=sequence_length)
    _, y_test = make_sequences(test_arr, seq_length=sequence_length)

    naive_train = naive_baseline_from_df(train_df, sequence_length)
    naive_test = naive_baseline_from_df(test_df, sequence_length)

    true_train = scaler_y.inverse_transform(y_train)
    true_test = scaler_y.inverse_transform(y_test)

    if naive_train.shape != true_train.shape:
        raise ValueError("학습 구간 기준선과 타깃의 표본 수가 다릅니다.")

    if naive_test.shape != true_test.shape:
        raise ValueError("테스트 구간 기준선과 타깃의 표본 수가 다릅니다.")

    train_metrics = evaluate(
        naive_train,
        true_train,
        name="train naive",
        unit="원",
    )
    test_metrics = evaluate(
        naive_test,
        true_test,
        name="test naive",
        unit="원",
    )

    print(
        "학습 구간 naive 기준선: 단위={}, MAE={:.2f}, MAPE(%)={:.6f}, "
        "표본수={}, 제외표본수={}".format(
            train_metrics["unit"],
            train_metrics["mae"],
            train_metrics["mape"],
            train_metrics["n_samples"],
            train_metrics["n_skipped"],
        )
    )
    print(
        "테스트 구간 naive 기준선: 단위={}, MAE={:.2f}, MAPE(%)={:.6f}, "
        "표본수={}, 제외표본수={}".format(
            test_metrics["unit"],
            test_metrics["mae"],
            test_metrics["mape"],
            test_metrics["n_samples"],
            test_metrics["n_skipped"],
        )
    )
