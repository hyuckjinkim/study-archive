import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 시계열 데이터셋 클래스 정의
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_size, window_size, future_size, infer_mode):
        self.infer_mode = infer_mode
        
        self.x_list=[]
        self.y_list=[]
        i=0
        while True:
            if len(data)<i+sequence_size+future_size:
                break
            seq_x = data.iloc[i:i+sequence_size,:].values
            seq_y = data.iloc[i+sequence_size:i+sequence_size+future_size,-1].values
            i+=window_size
            self.x_list.append(torch.Tensor(seq_x))
            self.y_list.append(torch.Tensor(seq_y))
        

    def __getitem__(self, index):
        data  = self.x_list[index]
        label = self.y_list[index]
        if self.infer_mode == False:
            return data, label
        else:
            return data

    def __len__(self):
        return len(self.x_list)