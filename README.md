# Capston
파일을 잘못 올렸는지 브런치가 생겨서 2개로 나눠졌습니다

첫번째는 LSTM, GRU등등 모델들을 가지고 만든것

두번째는 SCINet인데 SCINet은 도저히 제가 따로 다시 분석해서 다시 만들기가 힘들어서 일단 예전에 수정한 파일들을 올렸습니다

Attention은 포기하려고 했었는데 메일 보고 나서 다시 찾아보고 만들어보았습니다

데이터는 파일내의 Upbit의 BTC데이터를 가져와서 사용하였습니다

2024-10-03 12:39:00 부터 2024-10-31 17:57:00 까지 총 13440개의 데이터








RevIN의 모델은 여기를 참고하였습니다

https://github.com/ts-kim/RevIN.git

@inproceedings{kim2021reversible,
  title     = {Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift},
  author    = {Kim, Taesung and 
               Kim, Jinhee and 
               Tae, Yunwon and 
               Park, Cheonbok and 
               Choi, Jang-Ho and 
               Choo, Jaegul},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://openreview.net/forum?id=cGDAkQo1C0p}
}

SCINet 모델 인용을 여기서 해왔습니다

https://github.com/cure-lab/SCINet.git

@article{liu2022SCINet,

title={SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction},

author={Liu, Minhao and Zeng, Ailing and Chen, Muxi and Xu, Zhijian and Lai, Qiuxia and Ma, Lingna and Xu, Qiang},

journal={Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), 2022},

year={2022}

}
