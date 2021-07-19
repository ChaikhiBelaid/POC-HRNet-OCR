import os
import argparse
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from datetime import datetime, time
#Configurations parser arguments :
parser = argparse.ArgumentParser(description='Visualizing results of HRNet')
parser.add_argument('--num_epoch', type=int, default=308,
                           help='The number of epoch trained until now')
parser.add_argument('--plotLoss', type=bool, default=False,
                           help='plot the losse infos function of all training')
parser.add_argument('--plotTime', type=bool, default=False,
                           help='plot the time values regarding epochs')
parser.add_argument('--plotMIoU', type=bool, default=False,
                           help='plot the MIoU and Best mIoU per epoch')
parser.add_argument('--display', type=bool, default=True,
                            help='Display infos about the training')
parser.add_argument('--Epoch', type=int, default=None,
                            help='Display infos about the epoch and plot')

# Open the file to read :
file0 = open('training_infos2.txt', 'r')
global_infos = file0.readlines()
file1 = open('training_infos3.txt','r')
global_infos1 = file1.readlines()

#List of classes :
List_classes = ['unlabeled',                 
'ego vehicle',               
'rectification border',      
'out of roi',                
'static',                    
'dynamic',                   
'ground',                    
'road',                      
'sidewalk',                  
'parking',                   
'rail track',                
'building',                  
'wall',                      
'fence',                     
'guard rail',                
'bridge',                    
'tunnel',                    
'pole',                      
'polegroup',                 
'traffic light',             
'traffic sign',              
'vegetation',                
'terrain',              
'sky',                  
'person',               
'rider',                
'car',                  
'truck',                
'bus',                  
'caravan',              
'trailer',              
'train',                
'motorcycle',           
'bicycle',              
'license plate',        
]
def extract_infos(global_infos, start_line, final_line):
    """
    Input :
        global_infos : list of lines of the file training_info.txt
        start_line : -int-, the number of the first line
        final_line : -int-, the number of the last line
    Outputs :
        losses : list of all losse values of a given epoch per iteration
        times : list of time values of a given epoch per iteration
        learning_rates : list of learning rate values of a given epoch per iteration
    """
    infos = global_infos[start_line:final_line]
    losses = [float(info.split(',')[-1].split(':')[-1]) for info in infos]
    times = [float(info.split(',')[1].split(':')[-1]) for info in infos]
    learning_rates = [float(info.split(',')[2].split(':')[-1].replace(']','').replace('[','')) for info in infos]
    return losses, times, learning_rates
def plot_infos(epoch, losses, times, learning_rates, time_now):
    """
    Inputs : 
        epoch : the number of the epoch
        losses : list of losse values regarding the iterations of a given epoch
        times : list of losse values of each iteration of a given epoch
        learning_rates : list of losse values used in each iteration of a given
        time_now : string containing the current time
    Outputs :
        plot losses, times and learning rates regarding iterations of a given epoch
    """
    path_to_save1 = "loss_time_epoch_"+str(epoch)+'_'+time_now+'.jpg'
    path_to_save2 = "lr_epoch_"+str(epoch)+'_'+time_now+'.jpg'
    plt.plot(np.linspace(0, 2980, 298), np.array(losses), 'r',label='losses')
    plt.plot(np.linspace(0, 2980, 298), np.array(times), 'b', label='times(s)')
    plt.xlabel('#number of iterations')
    plt.title(f'Evolution of losse, time values regarding iterations of epoch {epoch}')
    plt.legend()
    plt.savefig(path_to_save1)
    plt.show()
    plt.plot(np.linspace(0, 2980, 298), np.array(learning_rates), 'g')
    plt.xlabel('#number of iterations')
    plt.ylabel('learning rate values')
    plt.title(f'Evolution of learning rate values regarding iterations of epoch {epoch}')
    plt.savefig(path_to_save2)
    plt.show()

def table_infos(losses, times, learning_rates):
    """
    Inputs : outputs of extract_infos function
        losses : list of losse values regarding the iterations of a given epoch
        times : list of losse values of each iteration of a given epoch
        learning_rates : list of losse values used in each iteration of a given
    Outputs :
        df : dataframe of information of a given epoch
    """
    iterations = range(0, 2980, 10)
    data= {
            "Iteration": iterations,
            "Losse": losses,
            "Times(s)": times,
            "Learning rate": learning_rates
          }
    df = pd.DataFrame(data)
    return df
def result_infos(losses, times, learning_rates, important_infos):
    MIoU =  float(important_infos.split(',')[1].split(':')[-1])
    Best_MIoU =  float(important_infos.split(',')[-1].split(':')[-1])
    duration = sum(times)
    mean_loss = mean(losses)
    max_loss = max(losses)
    min_loss = min(losses)
    max_lr = max(learning_rates)
    min_lr = min(learning_rates)
    dict_infos = {
        "losses":losses,
        "times":times,
        "learning_rates":learning_rates,
        "MIoU":MIoU,
        "Best_MIoU":Best_MIoU,
        "Duration":duration,
        "mean_loss":mean_loss,
        "max_loss":max_loss,
        "min_loss":min_loss,
        "max_lr":max_lr,
        "min_lr":min_lr
        }
    return dict_infos

def infos_per_epoch(num_epoch):
    """
    Inputs :
        num_epoch : the number of epochs that we have in the training
    Outputs :
        infos_per_epoch : is a dictionnary containing all infos about the training
    """
    start_line = 120
    final_line = 418
    infos_per_epoch = {}
    importants_infos = {}
    for i in range(1, 342):
        epoch = i-1
        losses, times, learning_rates = extract_infos(global_infos, start_line, final_line)
        
        for line in global_infos[final_line+49:final_line+49+14] :
            if 'Loss:' in line and 'Iter:' not in line:
                index_imp_infos = global_infos.index(line)
        
                          
        for line in global_infos[final_line:index_imp_infos]:
            if '0 [' in line :
                index_BestMIoU_classes =  global_infos.index(line)
            if '1 [' in line :
                index_MIoU_classes = global_infos.index(line)
        
        BestMIoU_classes = ''.join(global_infos[index_BestMIoU_classes:index_MIoU_classes])
        BestMIoU_classes.replace('0 [','')
        BestMIoU_classes = ' '.join(BestMIoU_classes.split(' ')[:-1])
        BestMIoU_classes = BestMIoU_classes.split(' ')
        BestMIoU_classes.remove('0')
        BestMIoU_classes[0] = BestMIoU_classes[0].replace('[','') 
        BestMIoU_classes[-1] = BestMIoU_classes[-1].replace(']','')
        BestMIoU_classes = [0. if l=='' else float(l) for l in BestMIoU_classes] 

        MIoU_classes = '\n'.join(global_infos[index_MIoU_classes:index_imp_infos-1])
        MIoU_classes.replace('1','')
        MIoU_classes = ' '.join(MIoU_classes.split(' ')[:-1])
        MIoU_classes = MIoU_classes.split(' ')
        MIoU_classes.remove('1')
        MIoU_classes[0] = MIoU_classes[0].replace('[','') 
        MIoU_classes[-1] = MIoU_classes[-1].replace(']','') 
        MIoU_classes = [0. if l=='' else float(l) for l in MIoU_classes] 

        important_infos = global_infos[index_imp_infos]
        importants_infos['EPOCH '+str(epoch)] = (important_infos, epoch,start_line, final_line, index_imp_infos)
        dict_results = result_infos(losses, times, learning_rates, important_infos)
        dict_results['BestMIoU_classes']  = BestMIoU_classes
        dict_results['MIoU_classes'] = MIoU_classes 

        
         
        infos_per_epoch['EPOCH '+str(epoch)] = dict_results

        for line in global_infos[index_imp_infos:index_imp_infos+15]:
            if 'Epoch: ['+str(epoch)+'/484] Iter:[0/2975]' in line :
                start_line = global_infos.index(line)
        final_line = start_line+298
    start_line = 0
    final_line = 297
    for i in range(342, num_epoch):
        epoch = i-1
        losses, times, learning_rates = extract_infos(global_infos1, start_line, final_line)
        
        for line in global_infos1[final_line:final_line+14] :
            if 'Loss:' in line and 'Iter:' not in line:
                index_imp_infos = global_infos1.index(line)
        
                          
        for line in global_infos1[final_line:index_imp_infos]:
            if '0 [' in line :
                index_BestMIoU_classes =  global_infos1.index(line)
            if '1 [' in line :
                index_MIoU_classes = global_infos1.index(line)
        
        BestMIoU_classes = ''.join(global_infos1[index_BestMIoU_classes:index_MIoU_classes])
        BestMIoU_classes.replace('0 [','')
        BestMIoU_classes = ' '.join(BestMIoU_classes.split(' ')[:-1])
        BestMIoU_classes = BestMIoU_classes.split(' ')
        BestMIoU_classes.remove('0')
        BestMIoU_classes[0] = BestMIoU_classes[0].replace('[','') 
        BestMIoU_classes[-1] = BestMIoU_classes[-1].replace(']','')
        BestMIoU_classes = [0. if l=='' else float(l) for l in BestMIoU_classes] 

        MIoU_classes = '\n'.join(global_infos1[index_MIoU_classes:index_imp_infos-1])
        MIoU_classes.replace('1','')
        MIoU_classes = ' '.join(MIoU_classes.split(' ')[:-1])
        MIoU_classes = MIoU_classes.split(' ')
        MIoU_classes.remove('1')
        MIoU_classes[0] = MIoU_classes[0].replace('[','') 
        MIoU_classes[-1] = MIoU_classes[-1].replace(']','') 
        MIoU_classes = [0. if l=='' else float(l) for l in MIoU_classes] 

        important_infos = global_infos1[index_imp_infos]
        importants_infos['EPOCH '+str(epoch)] = (important_infos, epoch,start_line, final_line, index_imp_infos)

        dict_results = result_infos(losses, times, learning_rates, important_infos)
        dict_results['BestMIoU_classes']  = BestMIoU_classes
        dict_results['MIoU_classes'] = MIoU_classes 

        
         
        infos_per_epoch['EPOCH '+str(epoch)] = dict_results

        for line in global_infos1[index_imp_infos:index_imp_infos+15]:
            if 'Epoch: ['+str(epoch)+'/484] Iter:[0/2975]' in line :
                start_line = global_infos1.index(line)
        final_line = start_line+298
    return infos_per_epoch, importants_infos
def display_infos_epoch_and_plot(epoch, infos_per_epoch, time_now):
    """
    Inputs :
        epoch : the index of the given epoch
        infos_per_epoch : is a dictionnary containing all infos about the training from the function infos_per_epoch
    Outputs :
        printing all infos about the given epoch such as global infos, duration and mean_loss....
    """
    dict_infos = infos_per_epoch['EPOCH '+str(epoch)]
    losses, times, learning_rates = dict_infos['losses'],dict_infos['times'],dict_infos['learning_rates']
    df_global = table_infos(losses, times, learning_rates)
    data_mean_classes ={
        'classe name' : List_classes[:len(infos_per_epoch['EPOCH '+str(epoch)]["MIoU_classes"])] ,
        'IoU value' : infos_per_epoch['EPOCH '+str(epoch)]["MIoU_classes"]
    } 
    df_mean_classes = pd.DataFrame(data_mean_classes)
    data_Bestmean_classes = {
        'classe name' : List_classes[:len(infos_per_epoch['EPOCH '+str(epoch)]["BestMIoU_classes"])] ,
        'Best IoU value' : infos_per_epoch['EPOCH '+str(epoch)]["BestMIoU_classes"]
    }
    df_Bestmean_classes = pd.DataFrame(data_Bestmean_classes) 
    print('EPOCH NUMBER '+str(epoch))
    print('---------Global Infos--------------')
    print(df_global)
    print('----------- Time infos ------')
    print('duration of the epoch (s)')
    print(dict_infos['Duration'])
    print('----------- Loss Info ----------')
    print('the mean loss value')
    print(dict_infos['mean_loss'])
    print('------------')
    print('max loss')
    print(dict_infos['max_loss'])
    print('----------')
    print('min loss')
    print(dict_infos['min_loss'])
    print('----------- Learning rate Info ----------')
    print('max learning rate')
    print(dict_infos['max_lr'])
    print('----------')
    print('min leaning rate')
    print(dict_infos['min_lr'])
    print('----------')
    print('-----------Intersection over Union(IoU) metric infos------')
    print('------mean IoU infos-------')
    print('Data Frame of mean IoU classe values\n')
    print(df_mean_classes)
    print('------------------------')
    print('mean IoU (mIoU)')
    print(dict_infos['MIoU'])
    print('------ Best mean IoU infos-------')
    print('Data Frame of Best mean IoU classe values\n')
    print(df_Bestmean_classes)
    print('------------------------')
    print('Best mIoU')
    print(dict_infos['Best_MIoU'])

    plot_infos(epoch,losses, times, learning_rates, time_now)
    
def infos_to_dataframe(infos_per_epoch, num_epoch):
    """
    Inputs :
        infos_per_epoch : is a dictionnary containing all infos about the training from the function infos_per_epoch
        num_epoch : the number of epochs that we have in the training
    Outputs :
        data_infos : a dataframe containing all infos about the training 
        MIoUs : a list of mean IoU of each epoch
        Best_MIoUs : a list of Best mean IoU of each epoch
        durations : a list of duration of each epoch
    """
    epochs = range(num_epoch)
    MIoUs =  [infos_per_epoch["EPOCH "+str(i)]["MIoU"] for i in range(num_epoch)]
    Best_MIoUs =  [infos_per_epoch["EPOCH "+str(i)]["Best_MIoU"] for i in range(num_epoch)]
    durations = [infos_per_epoch["EPOCH "+str(i)]["Duration"] for i in range(num_epoch)]
    mean_losses = [infos_per_epoch["EPOCH "+str(i)]["mean_loss"] for i in range(num_epoch)]
    max_losses = [infos_per_epoch["EPOCH "+str(i)]["max_loss"] for i in range(num_epoch)]
    min_losses = [infos_per_epoch["EPOCH "+str(i)]["min_loss"] for i in range(num_epoch)]
    max_lrs = [infos_per_epoch["EPOCH "+str(i)]["max_lr"] for i in range(num_epoch)]
    min_lrs = [infos_per_epoch["EPOCH "+str(i)]["min_lr"] for i in range(num_epoch)]
    data = {
        "Epoch":epochs,
        "MIoU": MIoUs,
        "Best MIoU":Best_MIoUs,
        "Duration(s)":durations,
        "mean losse":mean_losses,
        "max_losse":max_losses,
        "min losse":min_losses,
        "max learning rate":max_lrs,
        "min learning rate":min_lrs
        }
    data_infos = pd.DataFrame(data)
    return data_infos, MIoUs, Best_MIoUs, durations
    
def plot_losse_infos_per_epoch(num_epoch, infos_per_epoch, time_now):
    """
    Inputs :
        infos_per_epoch : is a dictionnary containing all infos about the training from the function infos_per_epoch
        num_epoch : the number of epochs that we have in the training
    Outputs :
        ploting mean, max and min of losses per epoch 
    """
    mean_losses = [infos_per_epoch["EPOCH "+str(i)]["mean_loss"] for i in range(num_epoch)]
    max_losses = [infos_per_epoch["EPOCH "+str(i)]["max_loss"] for i in range(num_epoch)]
    min_losses = [infos_per_epoch["EPOCH "+str(i)]["min_loss"] for i in range(num_epoch)]
    X = range(num_epoch)
    X = np.array(X)
    path_to_save = "loss_infos_"+'_'+time_now+'.jpg'
    plt.plot(X, np.array(mean_losses), 'r',label='mean_losses')
    plt.plot(X, np.array(max_losses), 'b', label='max_losses')
    plt.plot(X, np.array(min_losses), 'g', label='min_losses')
    plt.xlabel('#number of epoch')
    plt.title('Evolution of loss values regarding epochs')
    plt.legend()
    plt.savefig(path_to_save)
    plt.show()
def plot_time_infos_per_epoch(num_epoch, infos_per_epoch, time_now):
    """
    Inputs :
        infos_per_epoch : is a dictionnary containing all infos about the training from the function infos_per_epoch
        num_epoch : the number of epochs that we have in the training
    Outputs :
        ploting duration per epoch
    """
    durations = [infos_per_epoch["EPOCH "+str(i)]["Duration"] for i in range(num_epoch)]
    X = range(num_epoch)
    X = np.array(X)
    path_to_save = "time_infos_"+'_'+time_now+'.jpg'
    plt.plot(X, np.array(durations), 'r')
    plt.xlabel('#number of epoch')
    plt.ylabel('time')
    plt.title('Evolution time values regarding epochs')
    plt.savefig(path_to_save)
    plt.show()
def plot_mIoU_infos_per_epoch(num_epoch, infos_per_epoch, time_now):
    """
    Inputs :
        infos_per_epoch : is a dictionnary containing all infos about the training from the function infos_per_epoch
        num_epoch : the number of epochs that we have in the training
    Outputs :
        ploting mean IoU and Best mean IoU per epoch
    """
    MIoUs =  [infos_per_epoch["EPOCH "+str(i)]["MIoU"] for i in range(num_epoch)]
    Best_MIoUs =  [infos_per_epoch["EPOCH "+str(i)]["Best_MIoU"] for i in range(num_epoch)]
    X = range(num_epoch)
    X = np.array(X)
    path_to_save = "mIoU_infos_"+'_'+time_now+'.jpg'
    plt.plot(X, np.array(MIoUs), 'r',label='mIoU')
    plt.plot(X, np.array(Best_MIoUs), 'b', label='Best MIoU')
    plt.xlabel('#number of epoch')
    plt.legend()
    plt.title('Evolution mIoU and Best mIoU values regarding epochs')
    plt.savefig(path_to_save)
    plt.show()
def display_global_training_infos(infos_per_epoch, num_epoch):
    """
    Inputs :
        epoch : the index of the given epoch
        infos_per_epoch : is a dictionnary containing all infos about the training from the function infos_per_epoch
    Outputs :
        printing all infos about the training such as global infos, duration and max MIoU, min MIoU, max Best MIoU, min Best MIoU.
    """
    data_infos, MIoUs, Best_MIoUs, durations = infos_to_dataframe(infos_per_epoch, num_epoch)
    duration = sum(durations)
    max_MIoU = max(MIoUs)
    min_MIoU = min(MIoUs)
    max_Best_MIoU = max(Best_MIoUs)
    min_Best_MIoU = min(Best_MIoUs)
    print('---------Global Infos--------------')
    print('Number of trained epochs until now ')
    print(num_epoch)
    print('---------------')
    print(data_infos)
    print('----------- Time infos ------')
    print('duration of the training(s)')
    print(duration)
    print('-----------Intersection over Union(IoU) metric infos------')
    print('max of mIoU')
    print(max_MIoU)
    print('-----------')
    print('min of mIoU')
    print(min_MIoU)
    print('---------')
    print('max of Best mIoU')
    print(max_Best_MIoU)
    print('-----------')
    print('min of Best mIoU')
    print(min_Best_MIoU)
    
if __name__ == "__main__":
    args = parser.parse_args()
    num_epoch = args.num_epoch
    now = datetime.now()
    time_now = now.strftime("%d-%m-%Y %H-%M")
    infos_per_epoch, importants_infos = infos_per_epoch(num_epoch+1)
    if args.display :
        display_global_training_infos(infos_per_epoch, num_epoch)
        print("importants infos : ", importants_infos)
    if args.plotLoss :
        plot_losse_infos_per_epoch(num_epoch, infos_per_epoch, time_now)
    if args.plotTime :
        plot_time_infos_per_epoch(num_epoch, infos_per_epoch, time)
    if args.plotMIoU :
        plot_mIoU_infos_per_epoch(num_epoch, infos_per_epoch, time_now)
    if args.Epoch is not None :
        epoch = args.Epoch
        display_infos_epoch_and_plot(epoch, infos_per_epoch, time_now)
    
    
    

