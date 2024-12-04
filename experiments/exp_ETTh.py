import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
from data_process.etth_data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_BTC_hour, Dataset_Custom, Dataset_Pred
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from models.SCINet import SCINet

class Exp_ETTh(Exp_Basic):
    #클래스 정의 및 초기화
    def __init__(self, args):
        super(Exp_ETTh, self).__init__(args)
        self.args = args
    #SCINet모델을 생성
    def _build_model(self):

        #다중변수와 ECL데이터를 사용하면 입력차원의 수를 321개로 설정 / ECL데이터의 변수가 321개
        if self.args.features == 'M' and self.args.data == 'ECL':
            in_dim = 321
        #단일변수 예측시 입력차원개수는 1개 / 변수의 개수가 1개
        elif self.args.features == 'S':
            in_dim = 1
        #다중변수 예측시 입력차원개수는 7개 / 변수의 개수가 7개
        elif self.args.features == 'M':
            in_dim = 7
        else:
            print('Error!')

        #SCINet 변수설정
        model = SCINet(
            #output_len 예측할 시퀀스의 길이
            output_len=self.args.pred_len,
            #input_len 입력시퀀스의 길이
            input_len=self.args.seq_len,
            #input_dim 데이터의 입력차원
            input_dim= in_dim,
            #hid_size 은닉층 크기
            hid_size = self.args.hidden_size,
            #num_stacks SCINet에서의 스택의 개수
            num_stacks=self.args.stacks,
            #numlevels SCINet에서의 모델의 레벨 개수
            num_levels=self.args.levels,
            #concat_len SCINet의 시퀀스 길이 연결에 대한 설정
            concat_len = self.args.concat_len,
            #groups 입력채널을 그룹으로 나눌 때 사용하는 그룹 수
            groups = self.args.groups,
            #kernel SCINet의 합성곱신경망 연산 커널 크기
            kernel = self.args.kernel,
            #dropout 드롭아웃 비율
            dropout = self.args.dropout,
            #단일 시점 예측 설정
            single_step_output_One = self.args.single_step_output_One,
            #포지셔널 인코딩 사용 여부
            positionalE = self.args.positionalEcoding,
            #SCINet의 수정버전을 사용할지 여부
            modified = True,
            #실험성정에 대한 추가여부
            args = self.args)
        return model.float()

    def _get_data(self, flag):
        #args 데이터 설정
        args = self.args
        #데이터셋 설정
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'BTCh1':Dataset_BTC_hour
        }
        Data = data_dict[self.args.data]
        #타임임베딩이 timeF이면 timeenc가 1이되어 시간정보를 인코딩하고 그렇지않다면 0이 된다
        timeenc = 0 if args.embed!='timeF' else 1
        #테스트모드
        #데이터셋을 섞지않고 마지막 미니배치를 버린다. 배치사이즈는 지정한사이즈, freq는 주로 시간단위
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        #예측모드
        #셔플 X, 마지막 미니배치 안버림, 배치사이즈 1, freq는 세분화 수준을 맞춘다, 데이터셋은 Dataset_Pred
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        #기본모드 train,val
        #데이터셋을 섞고 마지막 미니배치를 버립니다
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        #데이터 인스턴스 설정
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        print(flag, len(data_set))
        #데이터 로더 설정
        #배치사이즈, 셔플 여부, 데이터를 로드할때 스레드수, 마지막 배치를 버릴지 여부
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader
    #모델의 최적호 기법 선택
    def _select_optimizer(self):
        #Adam을 사용해 파라미터 학습률 조정
        #self.model.parameters(): 현재 모델의 모든 학습 가능한 파라미터를 옵티마이저에 전달합니다. 
        # 이 파라미터들은 Adam 옵티마이저를 통해 업데이트됩니다
        #
        #lr=self.args.lr: 학습률 (learning rate)을 self.args.lr로 설정하여 옵티마이저에 전달합니다. 
        #학습률은 모델의 업데이트 속도를 조절하며, 적절한 학습률을 설정하는 것이 훈련 성능에 매우 중요합니다
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    #손실함수 선택
    def _select_criterion(self, losstype):
        #mse 계산
        if losstype == "mse":
            criterion = nn.MSELoss()
        #mae 계산
        elif losstype == "mae":
            criterion = nn.L1Loss()
        #아무것도 안고를 경우 L1loss선택
        else:
            criterion = nn.L1Loss()
        return criterion

    #모델 검증
    def valid(self, valid_data, valid_loader, criterion):
        #검증모드로 변경
        self.model.eval()
        total_loss = []

        preds = []
    
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        #valid_loader를 통해 검증데이터를 배치단위로 가져옴
        #입력배치데이터를 사용해 모델의 예측값과 실제값을 가져온다
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, batch_x, batch_y)

            #단일 스택 모델 self.args.stacks == 1일경우 예측값과 실제값간의 손실을 계산하여 loss에 저장
            if self.args.stacks == 1:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

                #preds와 trues 리스트에 각각 예측값과 실제값 추가
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                #pred_scales.append(pred_scale.detach().cpu().numpy())
                #true_scales.append(true_scale.detach().cpu().numpy())

            #이중 스택 모델 stacks ==2일경우 pred와 mid를 각각 true와 비교한후 두 손실값의 합을 구함
            elif self.args.stacks == 2:
                loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(), true.detach().cpu())

                #preds, trues, mids 각각 예측값과 실제값 추가
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                #pred_scales.append(pred_scale.detach().cpu().numpy())
                #mid_scales.append(mid_scale.detach().cpu().numpy())
                #true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

        #각 배치별 loss를 total_loss에 저장하고 평균을 구함
            total_loss.append(loss)
        total_loss = np.average(total_loss)

        #스택이 1일때 예측값과 실제값을 적절한 형상으로 변환
        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)
            #pred_scales = np.array(pred_scales)
            #true_scales = np.array(true_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            #pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            #true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])

            #mae,mse,rmse,mape,mspe,corr을 metric함수를 사용하여 성능지표 계산
            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            #maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            #print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

        #stacks가 2일경우 이중스택일 경우
        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)
            #pred_scales = np.array(pred_scales)
            #true_scales = np.array(true_scales)
            #mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            #true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            #pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            #mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
            print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            #maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            print('mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            #print('mid --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            #maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('final --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            #print('final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))
        else:
            print('Error!')

        return total_loss

    def train(self, setting):
        #데이터셋 가져오기
        train_data, train_loader = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        #모델 체크포인트 디렉토리 생성
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        #훈련 진행상황 추적 및 텐서보드 시각화
        writer = SummaryWriter('event/run_ETTh/{}'.format(self.args.model_name))

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        #옵티마이저 가져오기, 손실함수 설정
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        #Auto mixed Precision 설정및 모델 초기화
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y)

                if self.args.stacks == 1:
                    loss = criterion(pred, true)
                elif self.args.stacks == 2:
                    loss = criterion(pred, true) + criterion(mid, true)
                else:
                    print('Error!')

                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    print('use amp')    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            with torch.no_grad():
                print('--------start to validate-----------')
                valid_loss = self.valid(valid_data, valid_loader, criterion)
                print('--------start to test-----------')
                test_loss = self.valid(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))
                
            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('test_loss', test_loss, global_step=epoch)

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)
            
        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
        
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, batch_x, batch_y)

            if self.args.stacks == 1:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                #pred_scales.append(pred_scale.detach().cpu().numpy())
                #true_scales.append(true_scale.detach().cpu().numpy())
            elif self.args.stacks == 2:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                #pred_scales.append(pred_scale.detach().cpu().numpy())
                #mid_scales.append(mid_scale.detach().cpu().numpy())
                #true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

        if self.args.stacks == 1:
            preds = np.array(preds)
            trues = np.array(trues)

            #pred_scales = np.array(pred_scales)
            #true_scales = np.array(true_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            #true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            #pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            #maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))
            #print('TTTT denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mses, maes, rmses, mapes, mspes, corrs))

            #result save


        elif self.args.stacks == 2:
            preds = np.array(preds)
            trues = np.array(trues)
            mids = np.array(mids)

            #pred_scales = np.array(pred_scales)
            #true_scales = np.array(true_scales)
            #mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            #true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
            #pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
            #mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
            print('test shape:', preds.shape, mids.shape, trues.shape)

            mae, mse, rmse, mape, mspe, corr = metric(mids, trues)
            #maes, mses, rmses, mapes, mspes, corrs = metric(mid_scales, true_scales)
            print('Mid --> normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            #maes, mses, rmses, mapes, mspes, corrs = metric(pred_scales, true_scales)
            print('TTTT Final --> denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse, mape, mspe, corr))

        else:
            print('Error!')

        folder_path = 'exp/ett_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        print('Test:mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(mse, mae, rmse,
                                                                                                       mape, mspe,
                                                                                                       corr))
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        if self.args.save:
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            #np.save(folder_path + 'pred_scales.npy', pred_scales)
            #np.save(folder_path + 'true_scales.npy', true_scales)

            np.savetxt(f'{folder_path}/pred.csv', preds[0], delimiter=",")
            np.savetxt(f'{folder_path}/true.csv', trues[0], delimiter=",")
            #np.savetxt(f'{folder_path}/pred_scales.csv', pred_scales[0], delimiter=",")
            #np.savetxt(f'{folder_path}/true_scales.csv', true_scales[0], delimiter=",")
        return mae, 0.0, mse, 0.0

    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().cuda()
        batch_y = batch_y.float()

        if self.args.stacks == 1:
            outputs = self.model(batch_x)
        elif self.args.stacks == 2:
            outputs, mid = self.model(batch_x)
        else:
            print('Error!')
        #if self.args.inverse:
        #    outputs_scaled = dataset_object.inverse_transform(outputs)
        #if self.args.stacks == 2:
        #    mid_scaled = dataset_object.inverse_transform(mid)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].cuda()
        #batch_y_scaled = dataset_object.inverse_transform(batch_y)

        if self.args.stacks == 1:
            return outputs, 0.0 , 0 , 0 , batch_y , 0.0
            #return outputs, outputs_scaled, 0,0, batch_y, batch_y_scaled
        elif self.args.stacks == 2:
            return outputs, 0.0, mid, 0.0, batch_y, 0.0
        else:
            print('Error!')
