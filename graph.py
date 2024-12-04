import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 데이터 불러오기

#inv_True
#data1 = np.load('C:\\Users\\danyj\\Desktop\\VSCode\\RevIN\\baselines\\SCINet\\exp\\ett_results\\SCINet_BTCh1_ftM_sl384_ll96_pl96_lr5e-05_bs32_hid0.5_s2_l4_dp0.5_invTrue_itr0_oursTrue_seed42\\pred.npy')
#data2 = np.load('C:\\Users\\danyj\\Desktop\\VSCode\\RevIN\\baselines\\SCINet\\exp\\ett_results\\SCINet_BTCh1_ftM_sl384_ll96_pl96_lr5e-05_bs32_hid0.5_s2_l4_dp0.5_invTrue_itr0_oursTrue_seed42\\true.npy')

#inv_False
data1 = np.load('C:\\Users\\danyj\\Desktop\\VSCode\\RevIN\\baselines\\SCINet\\exp\\ett_results\\SCINet_BTCh1_ftM_sl384_ll96_pl96_lr5e-05_bs32_hid0.5_s2_l4_dp0.5_invFalse_itr0_oursTrue_seed42\\pred.npy')
data2 = np.load('C:\\Users\\danyj\\Desktop\\VSCode\\RevIN\\baselines\\SCINet\\exp\\ett_results\\SCINet_BTCh1_ftM_sl384_ll96_pl96_lr5e-05_bs32_hid0.5_s2_l4_dp0.5_invFalse_itr0_oursTrue_seed42\\true.npy')
data1=data1[:,:,-1]
data2=data2[:,:,-1]

# 그래프 크기 설정
plt.figure(figsize=(12, 6))

# 실제값 그래프
plt.plot(data1[:, -1], label='pred', color='red', alpha=0.7)
plt.plot(data2[:, -1], label='True', color='blue', alpha=0.7)
# 그래프 제목 및 레이블 추가
plt.title('Pred vs True')
plt.xlabel('Time Steps')

# 범례 추가
plt.legend()

# 그래프 표시
plt.show()