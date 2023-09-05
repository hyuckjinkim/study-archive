import numpy as np
import pandas as pd
import time
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import collections

# https://stackoverflow.com/questions/34486642/what-is-the-currently-correct-way-to-dynamically-update-plots-in-jupyter-ipython
def live_plot(data_dict, figsize=(15,9), title='',**kwargs):
    #global plot_df
    plot_df = pd.DataFrame(data_dict)
    
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    sns.lineplot(x=range(len(plot_df)),y=plot_df['train_loss'],color='gray',label='Train Loss')
    sns.scatterplot(x=range(len(plot_df)),y=plot_df['train_loss'],hue=plot_df['is_best'],
                    palette=['gray','red'] ,legend=False,**kwargs)
    sns.lineplot(x=range(len(plot_df)),y=plot_df['test_loss'],color='black',label='Test Loss')
    sns.scatterplot(x=range(len(plot_df)),y=plot_df['test_loss'] ,hue=plot_df['is_best'],
                    palette=['black','red'],legend=False,**kwargs)
    plt.title(title)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=2,bbox_to_anchor=(1,1)) # the plot evolves to the right
    
    plt.tight_layout()
    plt.show()

def train(
    model, train_dataloader, test_dataloader,
    criterion, optimizer, scheduler=None, device='cpu', 
    num_epochs=100, metric_period=10,
    target_logscale=False, early_stopping=None,
    plot=True,
    **plot_kwargs
):
    is_best = 0
    best_test_loss = float('inf')
    best_model = None

    is_best_list = []
    plot_dict = collections.defaultdict(list)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = 0.0
        test_loss = 0.0

        # 학습
        model.train()
        tr_preds = []
        tr_trues = []
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            if target_logscale:
                outputs = torch.exp(outputs)
                y       = torch.exp(y)
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            tr_preds += outputs.detach().cpu().numpy().flatten().tolist()
            tr_trues += y      .detach().cpu().numpy().flatten().tolist()

        # 검증
        te_preds = []
        te_trues = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_dataloader):
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                if target_logscale:
                    outputs = torch.exp(outputs)
                    y       = torch.exp(y)
                
                loss = criterion(outputs, y)
                
                test_loss += loss.item()
                te_preds += outputs.detach().cpu().numpy().flatten().tolist()
                te_trues += y      .detach().cpu().numpy().flatten().tolist()

        train_loss /= len(train_dataloader)
        test_loss /= len(test_dataloader)
        
        # #MSE -> RMSE
        # train_loss = np.sqrt(train_loss)
        # test_loss  = np.sqrt(test_loss)
        
        if scheduler is not None:
            scheduler.step(test_loss)

        # test_loss가 더 좋아졌을 경우 모델 저장 및 업데이트
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model
            torch.save(model.state_dict(), './mc/best_model.pt')
            is_best=1
        else:
            is_best=0
            
        epoch_end_time = time.time()
        if plot:
            epoch_run_time = epoch_end_time - epoch_start_time
            total_run_time = epoch_run_time*(num_epochs-(epoch+1))
            str_epoch = str(epoch+1).zfill(len(str(num_epochs)))
            
            plot_dict['train_loss'].append(train_loss)
            plot_dict['test_loss'] .append(test_loss)
            plot_dict['is_best']   .append(is_best)
            title = '[TensorBoard]\nEpoch: {} / {}\nTrain Loss: {:.4f}, Test Loss: {:.4f}\nElapsed: {:.2f}s, Total: {:.2f}s'\
                .format(str_epoch,num_epochs,train_loss,test_loss,epoch_run_time,total_run_time)
            live_plot(plot_dict,title=title,**plot_kwargs)
        else:
            if (epoch+1)%metric_period==0:
                epoch_run_time = epoch_end_time - epoch_start_time
                total_run_time = epoch_run_time*(num_epochs-(epoch+1))
                str_epoch = str(epoch+1).zfill(len(str(num_epochs)))
                
                progress_text = '{}[{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, elapsed: {:.2f}s, total: {:.2f}s'\
                    .format(np.where(is_best==1,'*',' '), str_epoch, num_epochs, 
                            train_loss, test_loss,epoch_run_time,total_run_time)
                print(progress_text)
            
        # early stopping 여부를 체크. 현재 과적합 상황 추적
        if early_stopping is not None:
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                break

    # 모든 epoch이 끝나면 최종 모델 저장
    torch.save(model.state_dict(), './mc/final_model.pt')
    print('Final model saved.')

    return best_model