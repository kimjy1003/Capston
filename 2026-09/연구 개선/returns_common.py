# -*- coding: utf-8 -*-
"""수익률 타깃 · 방향 정확도 분석 공통 모듈.

가격 수준 회귀는 구조적으로 persistence(직전 값)로 수렴한다. 여기서는 타깃을
1스텝(3분) 앞 로그수익률로 바꾸고 "방향을 맞히는가"를 본다.

핵심 원칙 4가지:
  1. 입력도 정상성 있는 형태(수익률·비율)로 바꾼다. 가격 수준을 그대로 넣으면
     MinMax의 분포 이동 문제가 입력 쪽에서 다시 생긴다.
  2. 스케일러는 실제 학습(fit) 구간에만 fit한다. 검증·테스트에는 transform만 한다.
  3. 방향 정확도는 50%가 아니라 단순 규칙 기준선과 비교한다.
     middle=(high+low)/2 는 봉 안의 평균 같은 값이라 수익률에 기계적 자기상관(+0.3)이
     생기고, 규칙 하나로 방향을 72% 맞힌다. 이 규칙을 못 넘으면 학습한 것이 없다.
  4. 테스트 표본은 가격 수준 분석(common.py)과 같은 3,743개로 맞춘다.
     타깃 시점이 같으므로 가격 MAE를 기존 표의 naive 기준선과 직접 비교할 수 있다.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# 이 폴더는 2026-09/연구 개선/ 이다. 공통 모듈과 데이터는 상위 폴더(2026-09)에 있다.
# 실행 위치와 상관없이 찾도록 파일 위치 기준으로 경로를 잡는다.
PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import common  # noqa: E402

DATA_PATH = PARENT_DIR / "data" / "BTC_3min_2024-10.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 원래 입력 8개(open, high, low, close, volume, value, ma_3, ma_5)를 정상성 있는 형태로 바꾼 것.
FEATURE_COLUMNS = [
    "r_close",   # log(c_t / c_{t-1})        종가 수익률
    "r_middle",  # log(m_t / m_{t-1})        middle 수익률
    "hl_range",  # log(h_t / l_t)            봉 폭
    "close_pos", # log(c_t / m_t)            봉 안에서 종가 위치 (middle 기계적 신호의 원천)
    "body",      # log(c_t / o_t)            몸통
    "gap",       # log(o_t / c_{t-1})        직전 종가 대비 시가
    "ma3_dev",   # log(c_t / ma3_t)
    "ma5_dev",   # log(c_t / ma5_t)
    "log_volume",  # log(1 + volume)
    "vwap_dev",  # log((value/volume) / m_t)  거래대금 가중 평균가의 위치
]

TARGETS = ("middle", "close")

# Upbit KRW 마켓 수수료 0.05% (매수·매도 각각).
UPBIT_FEE = 0.0005


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """common.load_data 결과에 정상성 입력 피처를 붙인다.

    행 수를 바꾸지 않는다(첫 행은 직전 값이 없어 NaN). 그래야 common.split_data와
    같은 경계로 나뉘어 가격 수준 분석과 테스트 표본이 일치한다.
    모든 피처는 t 시점까지의 값만 쓴다.
    """
    o, h, l, c = (df[col].astype(float) for col in ("open", "high", "low", "close"))
    m = df["middle"].astype(float)
    volume = df["volume"].astype(float)
    value = df["value"].astype(float)

    vwap = (value / volume).where(volume > 0, m)

    out = pd.DataFrame(index=df.index)
    out["r_close"] = np.log(c).diff()
    out["r_middle"] = np.log(m).diff()
    out["hl_range"] = np.log(h / l)
    out["close_pos"] = np.log(c / m)
    out["body"] = np.log(c / o)
    out["gap"] = np.log(o / c.shift(1))
    out["ma3_dev"] = np.log(c / df["ma_3"])
    out["ma5_dev"] = np.log(c / df["ma_5"])
    out["log_volume"] = np.log1p(volume)
    out["vwap_dev"] = np.log(vwap / m)

    out["middle"] = m
    out["close"] = c
    return out


def split_frames(
    features: pd.DataFrame,
    train_ratio: float = 0.7,
    fit_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """노트북과 같은 규칙으로 학습(fit) / 검증 / 테스트를 나눈다."""
    train_df, test_df = common.split_data(features, train_ratio=train_ratio)
    fit_df, val_df = common.split_data(train_df, train_ratio=fit_ratio)
    return fit_df, val_df, test_df


def fit_feature_scaler(fit_df: pd.DataFrame) -> StandardScaler:
    """학습(fit) 구간으로만 입력 스케일러를 적합한다."""
    values = fit_df[FEATURE_COLUMNS].dropna().to_numpy(dtype=float)
    return StandardScaler().fit(values)


def make_windows(
    part: pd.DataFrame,
    seq_length: int,
    target: str,
    scaler: StandardScaler,
) -> Dict[str, np.ndarray]:
    """입력 윈도와 1스텝 앞 로그수익률 타깃을 만든다.

    정렬 규칙은 common.make_sequences와 같다. 윈도 i는 i..i+seq_length-1 행이고
    타깃 시점은 i+seq_length. 수익률 타깃은 log p[i+seq] - log p[i+seq-1] 이다.
    NaN이 들어간 윈도(데이터 맨 앞)는 버린다.
    """
    if target not in TARGETS:
        raise ValueError("target은 middle 또는 close여야 합니다.")

    raw = part[FEATURE_COLUMNS].to_numpy(dtype=float)
    scaled = raw.copy()
    finite_rows = np.isfinite(raw).all(axis=1)
    scaled[finite_rows] = scaler.transform(raw[finite_rows])

    price = part[target].to_numpy(dtype=float)
    middle = part["middle"].to_numpy(dtype=float)
    close = part["close"].to_numpy(dtype=float)

    n = len(part) - seq_length
    if n <= 0:
        raise ValueError("seq_length보다 행 수가 커야 합니다.")

    keep = np.array([finite_rows[i : i + seq_length].all() for i in range(n)])
    idx = np.nonzero(keep)[0]
    last = idx + seq_length - 1
    nxt = idx + seq_length

    X = np.stack([scaled[i : i + seq_length] for i in idx]).astype(np.float32)
    return {
        "X": X,
        "y": np.log(price[nxt] / price[last]),
        "base_price": price[last],
        "next_price": price[nxt],
        # 규칙 기준선과 매매 평가용 원 단위 값
        "last_r_target": np.log(price[last] / price[last - 1]),
        "last_close_pos": np.log(close[last] / middle[last]),
        "next_r_close": np.log(close[nxt] / close[last]),
        "time": part.index[nxt].to_numpy(),
    }


# ---------------------------------------------------------------- 기준선

def majority_direction(y_fit: np.ndarray) -> int:
    """학습(fit) 구간에서 더 많았던 방향(+1/-1). 0 수익률은 세지 않는다."""
    return 1 if (y_fit > 0).sum() >= (y_fit < 0).sum() else -1


def rule_signals(w: Dict[str, np.ndarray], majority: int) -> Dict[str, np.ndarray]:
    """단순 규칙 기준선의 방향 신호. 신호가 0이면 학습 구간 다수 방향으로 채운다."""
    def fill(sig: np.ndarray) -> np.ndarray:
        sig = np.sign(sig).astype(int)
        sig[sig == 0] = majority
        return sig

    n = len(w["y"])
    return {
        "항상 다수 방향": np.full(n, majority, dtype=int),
        "직전 방향 유지": fill(w["last_r_target"]),
        "직전 방향 반대": fill(-w["last_r_target"]),
        "봉 내 위치 sign(close-middle)": fill(w["last_close_pos"]),
    }


# ---------------------------------------------------------------- 지표

def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """이항 비율의 Wilson 95% 신뢰구간."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def direction_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """방향 정확도. 실제 수익률이 정확히 0인 표본은 방향이 없으므로 뺀다."""
    pred_sign = np.sign(np.asarray(pred, dtype=float)).ravel()
    true_sign = np.sign(np.asarray(y, dtype=float)).ravel()
    mask = true_sign != 0
    hit = pred_sign[mask] == true_sign[mask]
    k, n = int(hit.sum()), int(mask.sum())
    lo, hi = _wilson(k, n)
    return {
        "accuracy": k / n if n else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "p_vs_half": float(stats.binomtest(k, n, 0.5).pvalue) if n else float("nan"),
        "n": n,
        "n_zero_excluded": int((~mask).sum()),
        # 한쪽 방향만 찍는 모델은 정확도가 다수 방향 비율로 수렴한다. 그걸 드러내기 위한 값.
        "pred_up_ratio": float((pred_sign[mask] > 0).mean()) if n else float("nan"),
    }


def mcnemar_exact(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """같은 표본에서 두 방향 예측의 적중 차이를 McNemar 정확 검정으로 본다."""
    true_sign = np.sign(y).ravel()
    mask = true_sign != 0
    a = np.sign(pred_a).ravel()[mask] == true_sign[mask]
    b = np.sign(pred_b).ravel()[mask] == true_sign[mask]
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    discordant = only_a + only_b
    p = float(stats.binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def regression_metrics(
    pred_r: np.ndarray,
    w: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """수익률 공간 지표와, 가격으로 되돌린 MAE의 naive 대비 배수."""
    pred_r = np.asarray(pred_r, dtype=float).ravel()
    y = w["y"]
    err = pred_r - y

    pred_price = w["base_price"] * np.exp(pred_r)
    price_mae = float(np.mean(np.abs(pred_price - w["next_price"])))
    naive_mae = float(np.mean(np.abs(w["base_price"] - w["next_price"])))

    # 예측이 상수이면 순위상관이 정의되지 않는다.
    if np.std(pred_r) > 0:
        spearman = stats.spearmanr(pred_r, y)
        ic, ic_p = float(spearman.statistic), float(spearman.pvalue)
    else:
        ic, ic_p = float("nan"), float("nan")
    return {
        "mae_bp": float(np.mean(np.abs(err)) * 1e4),
        "rmse_bp": float(np.sqrt(np.mean(err ** 2)) * 1e4),
        # 수익률 0 예측 = 가격 persistence. 1보다 작아야 persistence를 이긴 것.
        "mae_ratio_vs_zero": float(np.mean(np.abs(err)) / np.mean(np.abs(y))),
        "rmse_ratio_vs_zero": float(np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(y ** 2))),
        "ic_spearman": ic,
        "ic_pvalue": ic_p,
        "price_mae": price_mae,
        "price_mae_ratio_vs_naive": price_mae / naive_mae,
    }


def long_flat_backtest(
    pred: np.ndarray,
    next_r_close: np.ndarray,
    fee: float = UPBIT_FEE,
) -> Dict[str, float]:
    """상승 예측이면 종가에 보유, 아니면 현금. Upbit 현물이라 공매도는 없다.

    t 봉 종가에서 신호를 보고 종가에 진입·청산한다고 가정한다(호가 스프레드·슬리피지 무시).
    포지션이 바뀔 때마다 수수료를 뗀다. 로그수익률 합으로 누적한다.
    """
    position = (np.sign(np.asarray(pred, dtype=float)).ravel() > 0).astype(float)
    trades = np.abs(np.diff(np.concatenate(([0.0], position, [0.0]))))
    gross = float(np.sum(position * next_r_close))
    cost = float(trades.sum() * -np.log(1 - fee))
    return {
        "gross_log_return": gross,
        "net_log_return": gross - cost,
        "n_trades": int(trades.sum()),
        "exposure": float(position.mean()),
        "buy_hold_log_return": float(np.sum(next_r_close)),
    }


def fee_hurdle(y_close: np.ndarray, fee: float = UPBIT_FEE) -> Dict[str, float]:
    """3분 수익률 크기와 왕복 수수료를 bp로 비교한다.

    평균 |r|이 왕복 수수료보다 작으면, 매 봉 들어갔다 나오는 전략은
    방향을 100% 맞혀도 수수료를 메우지 못한다.
    """
    return {
        "mean_abs_close_bp": float(np.mean(np.abs(y_close)) * 1e4),
        "round_trip_fee_bp": float(2 * -np.log(1 - fee) * 1e4),
    }


def rows_to_markdown(rows: List[Dict[str, str]], columns: List[str]) -> str:
    """외부 의존성 없이 마크다운 표를 만든다."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"
