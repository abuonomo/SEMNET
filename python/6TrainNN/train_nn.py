import torch
from torch import nn
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt


class ff_network(nn.Module):

    def __init__(self):
        """
        Fully Connected layers
        """
        super(ff_network, self).__init__()

        self.semnet = nn.Sequential(
            nn.Linear(17, 50), # 17 properties
            nn.ReLU(),
            nn.Linear(50, 10), # very small network for tests
            nn.ReLU(),            
            nn.Linear(10, 1)
        )


    def forward(self, x):
        """
        Pass throught network
        """
        res = self.semnet(x)

        return res




def train_model(data_train0, data_test0, data_train1, data_test1, lr_enc, batch_size):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_semnet = ff_network().to(device)    
    optimizer_predictor = torch.optim.Adam(model_semnet.parameters(), lr=lr_enc)
    
    data_train0=torch.tensor(data_train0, dtype=torch.float).to(device)
    data_test0=torch.tensor(data_test0, dtype=torch.float).to(device)
    
    
    data_train1=torch.tensor(data_train1, dtype=torch.float).to(device)
    data_test1=torch.tensor(data_test1, dtype=torch.float).to(device)

    test_loss_total=[]
    for epoch in range(500): # should be much larger, with good early stopping criteria
        small_train_len=min(len(data_train0),len(data_train1))
        for batch_iteration in range(int(small_train_len/batch_size)): 
            idx = torch.randint(0, len(data_train0), (batch_size,))
            data_train0_samples = data_train0[idx]

            idx = torch.randint(0, len(data_train1), (batch_size,))
            data_train1_samples = data_train1[idx]

            calc_properties0 = model_semnet(data_train0_samples)
            criterion = torch.nn.MSELoss()
            curr_pred=torch.tensor([0] * batch_size, dtype=torch.float).to(device)
            real_loss0 = criterion(calc_properties0, curr_pred)
            loss0 = torch.clamp(real_loss0, min = 0., max = 50000.).double()

            calc_properties1 = model_semnet(data_train1_samples)

            criterion = torch.nn.MSELoss()
            curr_pred=torch.tensor([1] * batch_size, dtype=torch.float).to(device)
            real_loss1 = criterion(calc_properties1, curr_pred)
            loss1 = torch.clamp(real_loss1, min = 0., max = 50000.).double()

            loss=loss0+loss1
            optimizer_predictor.zero_grad()
            loss.backward()
            optimizer_predictor.step()

        # calculate train set
        model_semnet.eval()
        
        calc_properties0 = model_semnet(data_train0[1:3000])        
        criterion = torch.nn.MSELoss()

        curr_pred=torch.tensor([0] * len(data_train0), dtype=torch.float).to(device)

        real_loss0 = criterion(calc_properties0, curr_pred)

        real_loss_train_num0=real_loss0.detach().cpu().numpy()
        
        calc_properties1 = model_semnet(data_train1)        
        criterion = torch.nn.MSELoss()
        curr_pred=torch.tensor([1] * len(data_train1), dtype=torch.float).to(device)
        real_loss1 = criterion(calc_properties1, curr_pred)
        real_loss_train_num1=real_loss1.detach().cpu().numpy()        

        # calculate test set
        calc_properties0 = model_semnet(data_test0[1:3000])
        criterion = torch.nn.MSELoss()
        curr_pred=torch.tensor([0] * len(data_test0), dtype=torch.float).to(device)
        real_loss0 = criterion(calc_properties0, curr_pred)
        real_loss_test_num0=real_loss0.detach().cpu().numpy()
        
        calc_properties1 = model_semnet(data_test1)
        criterion = torch.nn.MSELoss()
        curr_pred=torch.tensor([1] * len(data_test1), dtype=torch.float).to(device)
        real_loss1 = criterion(calc_properties1, curr_pred)
        real_loss_test_num1=real_loss1.detach().cpu().numpy()      

        test_loss_total.append(np.mean(real_loss_test_num0)+np.mean(real_loss_test_num1))

        if epoch%10==0:
            info_str='epoch: '+str(epoch)+' - totalloss: ',np.mean(real_loss_train_num0)+np.mean(real_loss_train_num1) ,'; l1/l2: '+str(np.mean(real_loss_train_num0))+'/'+str(np.mean(real_loss_train_num1))+'; total: ',np.mean(real_loss_test_num0)+np.mean(real_loss_test_num1) ,' ts1/ts2: '+str(np.mean(real_loss_test_num0))+'/'+str(np.mean(real_loss_test_num1))
            print('train_model: ',info_str)
        model_semnet.train()
        
        
        if len(test_loss_total)>30: # early stopping
            if (test_loss_total[-10]-test_loss_total[-1])<=0 and (test_loss_total[-11]-test_loss_total[-1])<=0 and (test_loss_total[-12]-test_loss_total[-1])<=0:
                break

    plt.plot(test_loss_total)
    plt.pause(0.05)
    plt.show

    return True
    
    

def train_nn_one_instance(data_0,data_1,model_semnet):
    batch_size=100
    lr_enc=0.01

    model_semnet.train()

    train_valid_test_size=[0.5, 0.5, 0.0]
    x = [i for i in range(len(data_0))]  # random shuffle input
    shuffle(x)
    data_0 = data_0[x]
    idx_traintest=int(len(data_0)*train_valid_test_size[0])
    data_train0=data_0[0:idx_traintest]    
    data_test0=data_0[idx_traintest:]
    
    x = [i for i in range(len(data_1))]  # random shuffle input
    shuffle(x)
    data_1 = data_1[x]
    idx_traintest=int(len(data_1)*train_valid_test_size[0])
    data_train1=data_1[0:idx_traintest]    
    data_test1=data_1[idx_traintest:]
    
    print('train_nn_one_instance - start training')
    bestval_no=train_model(data_train0, data_test0, data_train1, data_test1, lr_enc, batch_size)    

    print('train_nn_one_instance - Calculate ROC')
    
    
    return bestval_no



def calculate_ROC(data_0,data_1,model_semnet):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_semnet.eval()
    data_0=torch.tensor(data_0, dtype=torch.float).to(device)
    data_1=torch.tensor(data_1, dtype=torch.float).to(device)
    future_data_0=model_semnet(data_0).detach().cpu().numpy()
    future_data_1=model_semnet(data_1).detach().cpu().numpy()
    
    print(future_data_0.shape)
    print(future_data_1.shape)
    
    all_pred=np.concatenate([future_data_0,future_data_1])
    corr_answers=np.array([0] * len(future_data_0) + [1] * len(future_data_1))
    idx1=np.flip(np.argsort(all_pred,axis=0)) # indices from strongest predicted connection to shortest one
    
    print(corr_answers.shape)
    corr_answers_sorted=corr_answers[idx1]
    
    xpos=[0]
    ypos=[0]
    ROC_vals=[]
    for ii in range(len(corr_answers_sorted)):
        if corr_answers_sorted[ii]==1:
            xpos.append(xpos[-1])
            ypos.append(ypos[-1]+1)
        if corr_answers_sorted[ii]==0:
            xpos.append(xpos[-1]+1)
            ypos.append(ypos[-1])      
            ROC_vals.append(ypos[-1])
    
    ROC_vals=np.array(ROC_vals)/max(ypos)
    ypos=np.array(ypos)/max(ypos)
    xpos=np.array(xpos)/max(xpos)
    
    
    plt.plot(xpos, ypos)
    plt.show()
        
    AUC=sum(ROC_vals)/len(ROC_vals)
    print('Area Under Curve: ', AUC,'\n\n\n')
            
    return future_data_0, future_data_1   



def train_nn(all_data_0, all_data_1, prediction_distance, start_year):
    for y in range(len(all_data_0[1:3])):
        print('train_nn - year ',start_year+y+2)  
    
        model_semnet = ff_network()
        train_nn_one_instance(all_data_0[y],all_data_1[y],model_semnet)
        print('train_nn - finished - year ',start_year+y+2)
        print('Calculate ROC & AUC')
        
        if y+prediction_distance<len(all_data_0):
            future_data_0, future_data_1 = calculate_ROC(all_data_0[y+prediction_distance],all_data_1[y+prediction_distance],model_semnet)
            
    return True 
