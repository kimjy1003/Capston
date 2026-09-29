# Capston — BTC 단기 시계열 예측

졸업논문으로 작성한 코드를 **후속연구를 위해 재작성**한 폴더다(2026년 9월).
원본 노트북은 저장소 최상위에 그대로 두었다.
원본은 수치를 신뢰할 수 없는 결함이 있었고, 그 결함을 고친 뒤 결론이 바뀌었다.

데이터는 Upbit KRW-BTC 3분봉, 2024-10-03 12:27 ~ 2024-10-31 17:57, 13,440행.
원본은 실행할 때마다 `pyupbit` API를 호출했으나, 지금은 `data/BTC_3min_2024-10.csv`로 고정해
네트워크 없이 재현된다. 재수집한 값은 원 논문 노트북에 찍혀 있던 출력과 소수점까지 일치한다.

## 결과

타깃은 `middle = (high + low) / 2`의 1스텝(3분) 앞 값. 전 모델 동일하게 테스트 표본 3,743개.
기준선은 **naive persistence**(직전 시점 값을 그대로 예측)다.

| 모델 | MAE(원) | RMSE(원) | MAPE(%) | naive대비MAE배수 | 판정 |
| --- | --- | --- | --- | --- | --- |
| naive 기준선 | 38,595.69 | 57,182.72 | 0.0401 | 1.000 | 기준선 |
| LSTM RevIN | 32,614.88 | 46,145.70 | 0.0339 | 0.845 | 기준선보다 나음 |
| GRU RevIN | 34,311.69 | 48,810.87 | 0.0357 | 0.889 | 기준선보다 나음 |
| GRU MinMax | 682,398.86 | 1,060,091.95 | 0.6874 | 17.681 | 기준선보다 나쁨 |
| biLSTM MinMax | 1,946,288.78 | 2,864,971.38 | 1.9663 | 50.428 | 기준선보다 나쁨 |
| LSTM MinMax | 2,297,356.52 | 3,422,245.34 | 2.3192 | 59.524 | 기준선보다 나쁨 |
| LSTM Attention | 2,485,686.17 | 3,404,780.12 | 2.5222 | 64.403 | 기준선보다 나쁨 |

**MinMax는 추세가 있는 가격 시계열에 구조적으로 쓸 수 없다.**
10월 한 달간 BTC가 8,200만원에서 1억원으로 올라, 학습 구간에 fit한 MinMax로 테스트 타깃을 변환하면
0.882 ~ 1.849 범위가 되고 그중 **92.8%가 1.0을 넘는다**. [0,1]만 출력하도록 학습된 모델은
그 값을 낼 방법이 없다. 원 논문이 MAPE 0.22%를 얻었던 것은 테스트셋에 스케일러를 다시 fit했기 때문이고,
그 유출을 제거하자 실제 성능이 드러났다.

**RevIN은 그 분포 이동 문제를 실제로 해결한다.** 윈도마다 채널별로 정규화하므로 학습 구간의
가격 범위에 묶이지 않고, 두 RevIN 모델만 naive 기준선을 이겼다.

## 재현

모든 명령과 노트북은 이 폴더(`2026-09/`)에서 실행한다. 데이터·결과 경로가 이 폴더 기준 상대경로다.
```
python common.py        # naive 기준선 수치 확인
```
노트북 6개를 순서 상관없이 실행한 뒤:
```
python make_report.py   # results/*.json 을 모아 최종 표 생성
```
`make_report.py`는 결과들의 테스트 표본 수가 하나라도 다르면 에러를 내고 멈춘다.
서로 다른 구간에서 잰 수치를 한 표에 올린 것이 원 논문의 핵심 결함이었으므로 코드로 막았다.

환경: Python 3.11, torch 2.6.0+cu124, scikit-learn, pandas 2.x, numpy 1.x.

## 구조

- `common.py` — 로드 / 분할 / 정규화 / 시퀀스 / 평가 공통 모듈. 노트북 6개가 전부 이걸 쓴다.
- `make_report.py` — 결과 종합 및 검증.
- `data/BTC_3min_2024-10.csv` — 고정된 원본 데이터.
- `results/` — 노트북별 결과 JSON과 최종 표.
- 노트북 6개 — LSTM/GRU × MinMax/RevIN, 양방향 LSTM, LSTM+Attention.

## 원본에서 고친 것

1. **스케일러를 테스트셋에 다시 fit하던 것**(`fit_transform`을 학습셋과 테스트셋에 각각 호출).
   학습 구간에만 fit하고 나머지는 transform만 하도록 바꿨다. 검증셋에도 fit하지 않는다.
2. **RevIN이 RevIN으로 동작하지 않던 것.** `(N, 9)` 2차원 텐서에 걸어서 `dim2reduce`가 빈 튜플이 되고
   행렬 전체가 스칼라 하나로 정규화되고 있었다(volume 약 2와 value 약 2억이 한 통계에 섞였다).
   모델 안에서 `(batch, seq, channel)`에 적용하도록 바꿨고, `revin.parameters()`가 optimizer에
   포함되지 않아 학습되지 않던 affine도 이제 학습된다(파라미터 16개).
   타깃 `middle`은 입력 8채널에 없으므로 그 윈도의 `middle` 평균/표준편차로 정규화하고 같은 통계로 되돌린다.
3. **양방향 LSTM의 타깃 누수.** `y = middle = (high+low)/2` 인데 `high`, `low`가 **같은 시점** 입력에
   들어 있었다. 예측이 아니라 항등식이었다. 시퀀스 길이도 1이라 되돌아볼 과거가 없었다.
   타깃을 `i+288` 시점으로 밀어 해결했다. 출력 ReLU도 제거했다(음수 출력에서 기울기가 죽는다).
4. **Attention 노트북에 학습/테스트 분할이 없던 것.** 전체 데이터에 `fit_transform`을 걸고
   `shuffle=True`로 전량 학습했으며 평가 지표 계산 자체가 없었다.
5. **"특성 중요도" 그래프 철회.** `FeatureAttention`의 softmax는 `hidden_dim`(64) 축에서 정규화되는데
   그 64개 중 앞 8개를 잘라 입력 피처 이름을 붙인 것이었다(출력이 `[1, 64, 1]`인 게 증거).
   은닉 유닛을 피처로 오인한 결과라 해석이 성립하지 않아 삭제했다.
   시간축 attention 그래프는 남기되 y축을 0부터 잡고 균일 기준선 `1/288`을 함께 그린다.
   원본은 `ylim(0.0165, 0.017)`로 폭 0.0005만 확대해 균일한 가중치를 구조처럼 보이게 했다.
6. **평가 조건 통일.** 원본은 MinMax가 테스트셋 `[1500:2000]` 500개 구간, RevIN이 테스트셋 전체,
   biLSTM이 학습 포함 전체 구간에서 지표를 냈고, GRU_RevIN5만 5분봉을 썼다.
   전부 3분봉 · 테스트셋 전체 · 동일 표본으로 맞췄다.
7. **naive 기준선 추가.** 이게 없으면 다른 어떤 개선도 좋아진 건지 알 수 없다.
8. **검증셋 · 조기 종료 · 시드 고정.** 학습 구간을 다시 8:2로 나눠(시계열 순서 유지) 검증셋을 만들고,
   검증 손실 기준 조기 종료(patience 15) 후 최적 가중치를 복원한다. seed 42.
   원본은 200~2000 epoch 고정에 검증셋도 시드도 없었다.

## 남은 일

- 타깃을 가격 수준에서 **수익률(차분)**로 바꾸고 **방향 정확도**를 지표에 추가하는 것.
  가격 레벨 회귀는 구조적으로 persistence로 수렴하므로, "얼마나 정확한가"보다
  "방향을 맞히는가"가 이 문제에 정직한 질문이다.
- Attention 모델만 `hidden_dim` 64 / 2층이라 다른 노트북(hidden 4 / 1층)보다 용량이 크다.
  아키텍처 효과와 용량 효과가 섞여 있어 그대로 해석할 수 없다.
- `master` 브랜치의 SCINet은 아직 손대지 않았다. 알려진 문제:
  `--inverse` 경로가 깨져 정규화값과 원본값을 비교하고(`experiments/exp_ETTh.py` 461~467행의
  `inverse_transform` 호출이 주석 처리됨), `Dataset_BTC_hour`가 ETT의 하드코딩 경계를 그대로 써서
  15,301행 중 뒤쪽 901행이 미사용이다.

## 인용

RevIN 구현은 아래를 참고했다. https://github.com/ts-kim/RevIN

```
@inproceedings{kim2021reversible,
  title     = {Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift},
  author    = {Kim, Taesung and Kim, Jinhee and Tae, Yunwon and Park, Cheonbok and Choi, Jang-Ho and Choo, Jaegul},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://openreview.net/forum?id=cGDAkQo1C0p}
}
```

SCINet(`master` 브랜치)은 아래를 인용했다. https://github.com/cure-lab/SCINet

```
@article{liu2022SCINet,
  title={SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction},
  author={Liu, Minhao and Zeng, Ailing and Chen, Muxi and Xu, Zhijian and Lai, Qiuxia and Ma, Lingna and Xu, Qiang},
  journal={Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), 2022},
  year={2022}
}
```
