
BTC_data_import.py로 데이터를 가져오고

run_ETTH.py파일에서 실행을 합니다

exp/ETT_checkpoints 에 훈련한 모델들이 저장되고

exp/ett_results에 모델들의 결과값들이 저장됩니다

graph.py파일로 저장한 결과값들을 불러와서 그래프로 그리기가 가능합니다

파일 실행 사용 예시

$env:CUDA_VISIBLE_DEVICES="0"

python -u run_ETTh.py --data BTCh1 --features M --seq_len 192 --label_len 96 --pred_len 96 --hidden-size 4 --stacks 1 --levels 3 --lr 0.009 --batch_size 16 --dropout 0.5 --ours --save --data_path 'BTCh1.csv' --root_path 'C:\Users\danyj\Desktop\VSCode\RevIN\baselines\SCINet\data\ETT'
