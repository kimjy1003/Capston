# -*- coding: utf-8 -*-
"""수익률 타깃 · 방향 정확도 실험.

타깃 2개 × 모델 3종을 같은 테스트 표본(3,743개)에서 비교한다.
  - 타깃 middle: 원 논문과 같은 가격. 거래할 수 없는 가격이고 기계적 자기상관이 있다.
  - 타깃 close : 실제로 거래할 수 있는 가격.
  - 모델: Ridge(선형, 최근 12스텝), LSTM, GRU (노트북과 같은 hidden 4 / 1층 / seq 288).
신경망은 시드마다 결과가 흔들리므로 시드 여러 개를 돌리고, 예측을 평균한 앙상블을 대표값으로 쓴다.

실행 (어느 폴더에서든 된다. 경로는 파일 위치 기준):
    python direction_experiment.py            # 전체 (시드 5개)
    python direction_experiment.py --quick    # 동작 확인용 (시드 1개, 3 epoch)
결과: 이 폴더의 results/direction_table.md / .csv (커밋),
      results/direction_summary.json, direction_predictions_*.npz (무시)
"""
import argparse
import copy
import json
import random
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import returns_common as rc  # 상위 폴더를 sys.path에 넣으므로 common보다 먼저 import한다
import common  # noqa: E402

# 노트북(LSTM_MinMax2 등)과 같은 설정
SEQ_LENGTH = 288
HIDDEN_SIZE = 4
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 20
MAX_EPOCHS = 200
PATIENCE = 15

RIDGE_LAGS = 12
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SEEDS = (42, 43, 44, 45, 46)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """난수 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RecurrentRegressor(nn.Module):
    """마지막 시점 은닉 상태로 다음 수익률 1개를 낸다."""

    def __init__(self, cell: str, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
        self.rnn = rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)


def _loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y.reshape(-1, 1)).float())
    # 시계열이므로 노트북과 같이 셔플하지 않는다.
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=DEVICE.type == "cuda")


def _run_epoch(loader, model, loss_fn, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if is_train:
                loss.backward()
                optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
    return total / count


def _predict(loader, model) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            outs.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(outs).ravel()


def train_recurrent(cell, seed, data, y_scaler, max_epochs) -> Dict:
    """검증 손실 기준 조기 종료로 학습하고 테스트 예측(원 단위 로그수익률)을 돌려준다."""
    set_seed(seed)
    fit_w, val_w, test_w = data["fit"], data["val"], data["test"]
    y_fit = y_scaler.transform(fit_w["y"].reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(val_w["y"].reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(test_w["y"].reshape(-1, 1)).ravel()

    fit_loader = _loader(fit_w["X"], y_fit)
    val_loader = _loader(val_w["X"], y_val)
    test_loader = _loader(test_w["X"], y_test)

    model = RecurrentRegressor(cell, fit_w["X"].shape[2], HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    model.apply(init_weights)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        _run_epoch(fit_loader, model, loss_fn, optimizer)
        val_loss = _run_epoch(val_loader, model, loss_fn)
        if val_loss < best_val:
            best_val, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)

    pred_scaled = _predict(test_loader, model)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return {
        "pred": pred,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        # 검증 MSE가 1(=타깃 분산)에 가까우면 사실상 평균만 예측하는 것이다.
        "best_val_mse_scaled": best_val,
        "seconds": time.time() - started,
    }


def train_ridge(data, y_scaler) -> Dict:
    """최근 RIDGE_LAGS 스텝을 펼친 선형 회귀. alpha는 검증셋 MSE로 고른다."""
    def flat(w):
        return w["X"][:, -RIDGE_LAGS:, :].reshape(len(w["X"]), -1)

    fit_w, val_w, test_w = data["fit"], data["val"], data["test"]
    y_fit = y_scaler.transform(fit_w["y"].reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(val_w["y"].reshape(-1, 1)).ravel()

    scores = {}
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(flat(fit_w), y_fit)
        scores[alpha] = float(np.mean((model.predict(flat(val_w)) - y_val) ** 2))
    best_alpha = min(scores, key=scores.get)
    model = Ridge(alpha=best_alpha).fit(flat(fit_w), y_fit)

    pred = y_scaler.inverse_transform(model.predict(flat(test_w)).reshape(-1, 1)).ravel()
    return {"pred": pred, "alpha": best_alpha, "val_mse_scaled": scores}


def prepare(target: str) -> Dict:
    """타깃별 윈도와 스케일러를 만든다."""
    features = rc.build_features(common.load_data(rc.DATA_PATH))
    fit_df, val_df, test_df = rc.split_frames(features)
    scaler = rc.fit_feature_scaler(fit_df)
    data = {
        name: rc.make_windows(part, SEQ_LENGTH, target, scaler)
        for name, part in (("fit", fit_df), ("val", val_df), ("test", test_df))
    }
    expected = len(test_df) - SEQ_LENGTH
    if len(data["test"]["y"]) != expected:
        raise ValueError(f"테스트 표본 수가 {expected}이 아닙니다. 가격 수준 분석과 비교할 수 없습니다.")
    return data


def evaluate_target(target: str, seeds, max_epochs: int) -> Dict:
    data = prepare(target)
    test_w = data["test"]
    y_scaler = StandardScaler().fit(data["fit"]["y"].reshape(-1, 1))

    # 규칙 기준선: 어떤 규칙이 최강인지는 학습(fit) 구간 정확도로 고른다. 테스트로 고르면 반칙이다.
    majority = rc.majority_direction(data["fit"]["y"])
    fit_rules = rc.rule_signals(data["fit"], majority)
    fit_acc = {name: rc.direction_metrics(sig, data["fit"]["y"])["accuracy"] for name, sig in fit_rules.items()}
    best_rule = max(fit_acc, key=fit_acc.get)
    test_rules = rc.rule_signals(test_w, majority)
    best_rule_signal = test_rules[best_rule]
    majority_signal = test_rules["항상 다수 방향"]

    entries: List[Dict] = []
    predictions: Dict[str, np.ndarray] = {}

    def add(name, kind, pred, extra=None, regression=True):
        predictions[f"pred_{len(entries):02d}"] = np.asarray(pred, dtype=float)
        entry = {
            "name": name,
            "kind": kind,
            "pred_key": f"pred_{len(entries):02d}",
            "direction": rc.direction_metrics(pred, test_w["y"]),
            # 가장 단순한 기준선. 학습 구간에서 고른 최강 규칙이 테스트에서 무너질 수 있으므로
            # 이 비교를 항상 함께 본다.
            "vs_majority": rc.mcnemar_exact(pred, majority_signal, test_w["y"]),
            "vs_best_rule": rc.mcnemar_exact(pred, best_rule_signal, test_w["y"]),
            "backtest": rc.long_flat_backtest(pred, test_w["next_r_close"]),
            "regression": rc.regression_metrics(pred, test_w) if regression else None,
        }
        entry.update(extra or {})
        entries.append(entry)
        d = entry["direction"]
        print(f"  {name:<32} 방향 {d['accuracy']*100:6.2f}%  예측상승 {d['pred_up_ratio']:.2f}", flush=True)

    print(f"\n[{target}] 최강 규칙(학습 구간 기준): {best_rule} ({fit_acc[best_rule]*100:.2f}%)")
    for name, sig in test_rules.items():
        add(name, "rule", sig, {"fit_accuracy": fit_acc[name]}, regression=False)

    # 수익률 0 예측(= 가격 persistence)은 방향이 없으므로 행으로 넣지 않는다.
    # 회귀표의 "/ 0예측", "/ naive" 배수가 곧 그 기준선 대비 값이다(기준선 자신은 1).

    ridge = train_ridge(data, y_scaler)
    add("Ridge (선형, 최근 12스텝)", "model", ridge["pred"], {"alpha": ridge["alpha"]})

    for cell in ("LSTM", "GRU"):
        runs = []
        for seed in seeds:
            run = train_recurrent(cell, seed, data, y_scaler, max_epochs)
            predictions[f"{cell.lower()}_seed{seed}"] = run["pred"]
            acc = rc.direction_metrics(run["pred"], test_w["y"])["accuracy"]
            print(f"    {cell} seed {seed}: 방향 {acc*100:.2f}%  best_epoch {run['best_epoch']}/{run['epochs_run']}"
                  f"  val_mse {run['best_val_mse_scaled']:.4f}  {run['seconds']:.0f}s", flush=True)
            runs.append({**run, "seed": seed, "accuracy": acc})
        ensemble = np.mean([r["pred"] for r in runs], axis=0)
        accs = np.array([r["accuracy"] for r in runs])
        add(f"{cell} (시드 {len(seeds)}개 앙상블)", "model", ensemble, {
            "seed_accuracy_mean": float(accs.mean()),
            "seed_accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "seed_runs": [{k: v for k, v in r.items() if k != "pred"} for r in runs],
        })

    y = test_w["y"]
    return {
        "target": target,
        "n_test": int(len(y)),
        "test_up_ratio": float((y > 0).sum() / (y != 0).sum()),
        "test_zero_ratio": float((y == 0).mean()),
        "majority_direction": majority,
        "best_rule": best_rule,
        "fee_hurdle": rc.fee_hurdle(test_w["next_r_close"]),
        "entries": entries,
        # JSON에는 넣지 않고 npz로 따로 저장한다(표를 재학습 없이 다시 만들 수 있게).
        "_arrays": {
            "y": test_w["y"],
            "next_r_close": test_w["next_r_close"],
            "time": test_w["time"].astype("datetime64[s]").astype("int64"),
            **predictions,
        },
    }


def _pct(x, digits=2):
    return f"{x * 100:.{digits}f}"


def build_tables(summaries: List[Dict]) -> str:
    parts = []
    for s in summaries:
        dir_rows, reg_rows = [], []
        for e in s["entries"]:
            d, b = e["direction"], e["backtest"]
            seed_col = ""
            if "seed_accuracy_mean" in e:
                seed_col = f"{_pct(e['seed_accuracy_mean'])} ± {_pct(e['seed_accuracy_std'])}"
            is_best = e["name"] == s["best_rule"]
            is_majority = e["name"] == "항상 다수 방향"
            dir_rows.append({
                "모델/기준선": e["name"] + (" (최강 규칙)" if is_best else ""),
                "방향정확도(%)": _pct(d["accuracy"]),
                "95% CI": f"{_pct(d['ci_low'], 1)} ~ {_pct(d['ci_high'], 1)}",
                "예측 상승비율": f"{d['pred_up_ratio']:.2f}",
                "다수방향 대비 p": "-" if is_majority else f"{e['vs_majority']['p_value']:.3g}",
                "최강규칙 대비 p": "-" if is_best else f"{e['vs_best_rule']['p_value']:.3g}",
                "시드별 정확도(%)": seed_col,
                "close 매매 수수료 전(%)": _pct(np.expm1(b["gross_log_return"])),
                "수수료 후(%)": _pct(np.expm1(b["net_log_return"])),
                "거래 횟수": b["n_trades"],
            })
            r = e["regression"]
            if r is not None:
                reg_rows.append({
                    "모델": e["name"],
                    "수익률 MAE(bp)": f"{r['mae_bp']:.3f}",
                    "MAE / 0예측": f"{r['mae_ratio_vs_zero']:.3f}",
                    "RMSE / 0예측": f"{r['rmse_ratio_vs_zero']:.3f}",
                    "IC(Spearman)": "-" if np.isnan(r["ic_spearman"]) else f"{r['ic_spearman']:+.3f}",
                    "IC p값": "-" if np.isnan(r["ic_pvalue"])
                    else ("<1e-10" if r["ic_pvalue"] < 1e-10 else f"{r['ic_pvalue']:.2g}"),
                    "가격 MAE / naive": f"{r['price_mae_ratio_vs_naive']:.3f}",
                })
        first = s["entries"][0]["backtest"]
        hurdle = s["fee_hurdle"]
        parts.append(
            f"### 타깃: {s['target']} 로그수익률 (테스트 {s['n_test']}개, "
            f"0 수익률 {_pct(s['test_zero_ratio'], 1)}% 제외 후 상승 비율 {_pct(s['test_up_ratio'], 1)}%)\n\n"
            + rc.rows_to_markdown(dir_rows, list(dir_rows[0].keys()))
            + "\n"
            + rc.rows_to_markdown(reg_rows, list(reg_rows[0].keys()))
            + "\n회귀 지표의 기준선은 수익률 0 예측(= 가격 persistence)이고, 배수가 1보다 작아야 이긴 것이다. "
            "타깃 middle의 가격 MAE / naive는 가격 수준 분석 표(final_table)의 naive대비MAE배수와 "
            "같은 표본·같은 정의다(close 타깃은 close 기준 naive).\n"
            + f"매매는 close 수익률로 평가(상승 예측이면 보유, 수수료 0.05% 편도). "
            f"같은 기간 단순 보유 {_pct(np.expm1(first['buy_hold_log_return']))}%, "
            f"3분 close 수익률의 평균 크기는 {hurdle['mean_abs_close_bp']:.1f}bp로 "
            f"왕복 수수료 {hurdle['round_trip_fee_bp']:.1f}bp"
            + ("보다 작다.\n" if hurdle["mean_abs_close_bp"] < hurdle["round_trip_fee_bp"] else "보다 크다.\n")
        )
    return "\n".join(parts)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="시드 1개, 3 epoch로 동작만 확인")
    args = parser.parse_args()

    seeds = SEEDS[:1] if args.quick else SEEDS
    max_epochs = 3 if args.quick else MAX_EPOCHS
    print(f"device={DEVICE}, seeds={list(seeds)}, max_epochs={max_epochs}")

    summaries = [evaluate_target(target, seeds, max_epochs) for target in rc.TARGETS]

    out_dir = rc.RESULTS_DIR
    out_dir.mkdir(exist_ok=True)
    suffix = "_quick" if args.quick else ""
    for s in summaries:
        np.savez_compressed(out_dir / f"direction_predictions_{s['target']}{suffix}.npz", **s.pop("_arrays"))
    (out_dir / f"direction_summary{suffix}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    table_md = build_tables(summaries)
    (out_dir / f"direction_table{suffix}.md").write_text(table_md, encoding="utf-8")

    flat_rows = []
    for s in summaries:
        for e in s["entries"]:
            row = {"target": s["target"], "name": e["name"], "kind": e["kind"]}
            row.update({f"dir_{k}": v for k, v in e["direction"].items()})
            row["vs_majority_p"] = e["vs_majority"]["p_value"]
            row["vs_best_rule_p"] = e["vs_best_rule"]["p_value"]
            row.update({f"bt_{k}": v for k, v in e["backtest"].items()})
            if e["regression"]:
                row.update({f"reg_{k}": v for k, v in e["regression"].items()})
            flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(out_dir / f"direction_table{suffix}.csv", index=False, encoding="utf-8-sig")

    print("\n" + table_md)


if __name__ == "__main__":
    main()
