import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from model import Generator,Combiner
from model import Discriminator,DiscriminatorM,DiscriminatorMST, DiscriminatorMZ,\
DiscriminatorMZR, Combiner,CombinerSeq,CombinerSeqL,CombinerSeqAtt,CombinerSeqAttReplace, GeneratorM,DiscriminatorM

from torch.autograd import Variable
from torchvision.utils import save_image
from FacialDataset import AFEWVA,AFEWVAReduced,SEWAFEWReduced, SEWAFEWReducedLatent
from utils import *
import time
import torch.nn.functional as F
import numpy as np
import torch
import datetime
from torchvision import transforms
from torch import nn
from calcMetrix import *
from config import *
import csv
import file_walker
import matplotlib.ticker as ticker
from PIL import Image
from scipy.special import softmax

import matplotlib.gridspec as gridspec

def str2bool(v):
    return v.lower() in ('true')
    

def train_only_comb_seq():
    
    #64,0,1200 32,1,2000? 32,2,
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-split', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-addLoss', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-singleTask', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-trainQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-dConv', nargs='?', const=1, type=int, default=64)#64 
    parser.add_argument('-batch_size', nargs='?', const=1, type=int, default=4000) #0 is ori, 1 is red 

    parser.add_argument('-sewa', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-useWeightNormalization', nargs='?', const=1, type=int, default=1)#0,1,2
    
    parser.add_argument('-useAll', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-seq_length', nargs='?', const=1, type=int, default=4)#1,2,4,8,16,32
    
    parser.add_argument('-use_attention', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-use_ch', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-use_h', nargs='?', const=1, type=int, default=1)#0,1,2
    
    parser.add_argument('-toLoad', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-toUpgrade', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-toAddAttention', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-numIters', nargs='?', const=1, type=int, default=200000)#0,1,2
    
    
    
    args = parser.parse_args()
    split = args.split
    addLoss = args.addLoss 
    singleTask = args.singleTask 
    isSewa = args.sewa 
    useWeight = args.useWeightNormalization
    useAll = args.useAll
    
    useAtt = args.use_attention
    useCH = args.use_ch
    useH = args.use_h
    
    trainQuadrant = args.trainQuadrant
    alterQuadrant = True
    
    per = args.per
    
    
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    image_size=128
    d_conv_dim=args.dConv
    lambda_cls=1
    lambda_rec=10
    lambda_gp=10
    inputC = 3#input channel for discriminator 
    
    
    toLoad = args.toLoad
    toUpgrade = args.toUpgrade
    toAddAttention = args.toAddAttention
    
    resume_iters=None #, help='resume training from this step') 
    
    num_iters=args.numIters #, help='number of total iterations for training D')
    num_iters_decay=100000 #, help='number of iterations for decaying lr')
    g_lr=0.0001 #, help='learning rate for G')
    d_lr=0.0001 #, help='learning rate for D')
    n_critic=5 #, help='number of D updates per each G update')
    beta1=0.5 #, help='beta1 for Adam optimizer')
    beta2=0.999 #, help='beta2 for Adam optimizer')
    
    isVideo = True
    toAlign = False
    seq_length = args.seq_length
    batch_size=int(truediv(args.batch_size,seq_length))#500, help='mini-batch size')
    
    # Test configuration.
    test_iters=200000 #, help='test model from this step')

    # Miscellaneous.
    num_workers=1
    mode='train' #, choices=['train', 'test'])
    use_tensorboard=False

    
    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples'
    result_dir='stargan/results'

    # Step size.
    log_step=10
    sample_step=1000
    model_save_step=10000
    lr_update_step=100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For fast training.
    cudnn.benchmark = True

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    #Split 
    #split = 0
    multi_gpu = False
    testSplit = split
    print("Test split " , testSplit)
    nSplit = 5
    listSplit = []
    for i in range(nSplit):
        if i!=testSplit : 
            listSplit.append(i)
    print(listSplit)
    
    if not isSewa : 
        main_name = 'AF-C-'
        d_name = 'AFEW-VA-Fixed'#'AFEW-VA-Fixed'
        dbType = 0
    else : 
        main_name = 'SE-C-'
        d_name = 'SEWA'
        dbType = 1
    
    if useH : 
        main_name += 'R-'
    
    if useCH : 
        main_name += 'CH-'
        
    load_to_add = main_name
    
    if useAtt : 
        main_name += 'A-'
        
    load_to_add_split = main_name
    
    
    
    mseLoss = nn.MSELoss()
    
    main_name+=(str(d_conv_dim)+'-')
    load_to_add+=(str(d_conv_dim)+'-')
    load_to_add_split+=(str(d_conv_dim)+'-')
    
    if trainQuadrant : 
        
        if alterQuadrant : 
            main_name+="-QDAL"
            c_dim = 1
        else :  
            main_name+="-QD"
            c_dim = 4
    
    save_name = main_name+str(testSplit)+'-n-'+str(seq_length)
    
    print('saving name is : ',save_name)
    
    load_to_add_split = load_to_add_split+str(testSplit)+'-n-'+str(seq_length)
    load_to_add = load_to_add+str(testSplit)+'-n-'+str(seq_length)
    load_prev = main_name+str(testSplit)+'-n-'+str(int(truediv(seq_length,2)))
    
    err_file = curDir+save_name+".txt"
    
    
    transform =transforms.Compose([
            transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
    ID = SEWAFEWReducedLatent([d_name], None, image_size, 1,split=True, nSplit = nSplit ,listSplit=listSplit
                ,isVideo=isVideo, seqLength = seq_length,dbType = dbType, returnQuadrant=trainQuadrant,
                returnWeight = useWeight,useAll = useAll, splitNumber=testSplit)
    
    dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True,worker_init_fn=worker_init_fn)

    VD = SEWAFEWReducedLatent([d_name], None, image_size, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
                ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant,dbType = dbType,useAll = useAll)
    
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)
   
    
    if not useH:
        model_ft = CombinerSeqAtt(image_size, d_conv_dim, c_dim, 4,64,512,seq_length,batch_size,useCH=useCH)
    else : 
        model_ft = CombinerSeqAttReplace(image_size, d_conv_dim, c_dim, 4,64,512,seq_length,batch_size,useCH=useCH)

    d_optimizer = torch.optim.Adam(model_ft.parameters(), d_lr, [beta1, beta2])
    print_network(model_ft, 'D')
    
    
    if toLoad:
        print('loading previous model ')
        model_ft.load_state_dict(torch.load(curDir+'t-models/'+save_name))
    elif toUpgrade : 
        print('upgrading from previous model ',load_prev)
        model_ft.load_state_dict(torch.load(curDir+'t-models/'+load_prev))
    elif toAddAttention : 
        print('adding attention to original model ',load_to_add)
        model_ft.load_state_dict(torch.load(curDir+'t-models/'+load_to_add))
    else : 
        model_ft.apply(weights_init_uniform_rule)
    
    model_ft.to(device)
    
    d_lr = d_lr

    start_iters = 0
    
    '''if resume_iters:
        start_iters = resume_iters
        restore_model(resume_iters)'''

    # Start training.
    print('Start training...')
    start_time = time.time()
    
    f = open(err_file,'w+')
    f.write("err : ")
    f.close()
    
    #best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 99999
    
    lMSA,lMSV,lCCV,lCCA,lICA,lICV,lCRA, lCRV, total = 9999,9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999
    
    w,wv,wa = None,None,None
    
    print('batch size : ',batch_size)
        
    for i in range(start_iters, num_iters):
        
        random.seed()
        manualSeed = random.randint(1, 10000) # use if you want new results
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        
        print('Epoch {}/{}'.format(i, num_iters - 1))
        print('-'*10)
        
        running_loss = 0
        model_ft.train()
            
        for x,(data) in enumerate(dataloader,0) : 
            
            rinputs_l, rlabels_l,rldmrk_l,_ = data[0],data[1],data[2],data[3]
            
            if useWeight : 
                w = data[5].cuda()
                
            ccPred_l = []
            
            model_ft.initialize(batch_size = rinputs_l.size(0)) #initialize for each seq
            prev_result = None 
            
            d_optimizer.zero_grad() 
            
            
            cumLoss = 0
            
            if useAtt : 
                l_h = []
                
            #print('shape of inputs',rinputs_l.shape)
            
            for y in range(seq_length): 
                
                rinputs, rlabels = rinputs_l[:,y].cuda(),rlabels_l[:,y].cuda()
                
                if useAtt : 
                    if len(l_h) > 0:
                        outputs = model_ft(rinputs,prev_h = l_h)
                    else : 
                        outputs = model_ft(rinputs)
                    if useCH : 
                        l_h.append(torch.cat((model_ft.lstm1_hdn[0][0],model_ft.lstm1_hdn[1][0]),1))
                    else : 
                        l_h.append(model_ft.lstm1_hdn[0][0])
                else :
                    outputs = model_ft(rinputs) 
                
                ccPred_l.append(outputs)
                
                #loss+=mseLoss(outputs,rlabels)
                
                loss = calcMSET(outputs,rlabels,w)
                cumLoss+=loss
                
                if addLoss :
                
                    ov,oa,lv,la = outputs[:,0],outputs[:,1], rlabels[:,0], rlabels[:,1]
                    
                    corV = -calcCORT(ov, lv, wv)
                    corA = -calcCORT(oa, la, wa)
                    
                    cccV = -calcCCCT(ov, lv, wv)
                    cccA = -calcCCCT(oa, la, wa)
                    
                    iccV = -calcICCT(ov, lv, wv)
                    iccA = -calcICCT(oa, la, wa)
                    
                    #<lossO =corV+corA +cccV+cccA+iccV+iccA
                    lossO = cccV+cccA+iccV+iccA
                
                if not addLoss : 
                    print("{}/{} loss : {}".format(x,int(len(dataloader.dataset)/batch_size),loss.item()))
                else : 
                    print("{}/{} loss : {:.8f}, cor : {:.8f}/{:.8f}, ccc : {:.8f}/{:.8f}, icc : {:.8f}/{:.8f}".format(x,int(len(dataloader.dataset)/batch_size),
                            loss.item(),corV.item(),corA.item(),cccV.item(),cccA.item(),iccV.item(),iccA.item()))
                
                f = open(err_file,'a')
                if not addLoss : 
                    f.write("{}/{} loss : {}\n".format(x,int(len(dataloader.dataset)/batch_size),loss.item()))
                else : 
                    f.write("{}/{} loss : {:.3f}, cor : {:.3f}/{:.3f}, ccc : {:.3f}/{:.3f}, icc : {:.3f}/{:.3f}\n".format(x,int(len(dataloader.dataset)/batch_size),
                            loss.item(),corV.item(),corA.item(),cccV.item(),cccA.item(),iccV.item(),iccA.item()))
                f.close()
                
                if addLoss : 
                    cumLoss += lossO
            
            
            cumLoss.backward()
            d_optimizer.step()
            
        
        
        # Decay learning rates.
        if (i+1) % lr_update_step == 0 and (i+1) > 50 : #(num_iters - num_iters_decay):
            d_lr -= (d_lr / float(num_iters_decay))
            update_lr(d_lr,d_optimizer)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            
        
        if i %2 == 0 : 
            if multi_gpu : 
                torch.save(model_ft.module.state_dict(),curDir+'t-models/'+save_name)
            else : 
                torch.save(model_ft.state_dict(),curDir+'t-models/'+save_name)
    
        #Deep copy the model_ft 
        if i%5 == 0 :#epoch_loss < lowest_loss : 
            lowest_loss = lowest_loss 
            
            model_ft.eval()    
            
            if True : 
            
                listValO = []
                listAroO = []
                
                listValL = []
                listAroL = []
                
                tvo = [];tao=[];tvl = []; tal = [];
                anyDiffer = False
               
                for x,(data) in enumerate(dataloaderV,0) :
                    
                    rinputs_l, rlabels_l,rldmrk_l = data[0],data[1],data[2]
                    
                    model_ft.initialize(rinputs_l.shape[0])
                    
                    if useAtt : 
                        l_h = []
                    
                    with torch.set_grad_enabled(False) : 
                        
                        pre_result = None
                        
                        for y in range(seq_length): 
                            rinputs, rlabels, rldmrk = rinputs_l[:,y], rlabels_l[:,y],rldmrk_l[:,y]
                     
                            inputs = rinputs.cuda()#to(device) 
                            labels = rlabels.cuda()#to(device)
                            
                            '''if useAtt : 
                                outputs,the_w = model_ft(inputs,ret_w=True)
                                print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                                print('w',the_w[:2])
                            else : 
                                outputs = model_ft(inputs)
                                print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                            '''
                            
                            if useAtt : 
                                if len(l_h) > 0:
                                    outputs,the_w = model_ft(inputs,prev_h = l_h,ret_w=True)
                                    print('w',the_w[:2])
                                else : 
                                    outputs = model_ft(inputs)
                                    
                                if useCH : 
                                    l_h.append(torch.cat((model_ft.lstm1_hdn[0][0],model_ft.lstm1_hdn[1][0]),1))
                                else : 
                                    l_h.append(model_ft.lstm1_hdn[0][0])
                            else :
                                outputs = model_ft(inputs) 
                                
                            #print('o shape',outputs.shape)
                            
                            print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                            
                            
                            
                            if outputs[:,0].shape[0] != batch_size : #in case the batch size is differ, usually at end of iter
                                anyDiffer = True 
                                print('differ')
                                tvo.append(outputs[:,0].detach().cpu())
                                tao.append(outputs[:,1].detach().cpu())
                                
                                tvl.append(labels[:,0].detach().cpu())
                                tal.append(labels[:,1].detach().cpu())
                            else :
                                print('equal')
                                listValO.append(outputs[:,0].detach().cpu())
                                listAroO.append(outputs[:,1].detach().cpu())
                                
                                listValL.append(labels[:,0].detach().cpu())
                                listAroL.append(labels[:,1].detach().cpu())
                        
                if len(listValO) > 0 : 
                    est_V = np.asarray(torch.stack(listValO)).flatten()
                    est_A = np.asarray(torch.stack(listAroO)).flatten()
                    
                    gt_V = np.asarray(torch.stack(listValL)).flatten()
                    gt_A = np.asarray(torch.stack(listAroL)).flatten()
                
                if anyDiffer : 
                    est_Vt = np.asarray(torch.stack(tvo)).flatten()
                    est_At = np.asarray(torch.stack(tao)).flatten()
                    
                    gt_Vt = np.asarray(torch.stack(tvl)).flatten()
                    gt_At = np.asarray(torch.stack(tal)).flatten()
                    
    #python main_red_test.py -useAll=1 -batch_size=6000 -seq_length=4 -use_attention=1 
    
     

    
    
                    #now concatenate
                    if len(listValO) > 0 : 
                        est_V = np.concatenate((est_V,est_Vt))
                        est_A = np.concatenate((est_A,est_At))
                        
                        gt_V = np.concatenate((gt_V,gt_Vt))
                        gt_A = np.concatenate((gt_A,gt_At))
                    else : 
                        est_V,est_A,gt_V,gt_A = est_Vt,est_At,gt_Vt,gt_At
                    
                print(est_V.shape, gt_V.shape)
                
                mseV = calcMSE(est_V, gt_V)
                mseA = calcMSE(est_A, gt_A)
                
                corV = calcCOR(est_V, gt_V)
                corA = calcCOR(est_A, gt_A)
                
                iccV = calcICC(est_V, gt_V)
                iccA = calcICC(est_A, gt_A)
                
                iccV2 = calcICC(gt_V, gt_V)
                iccA2 = calcICC(gt_A, gt_A)
                
                cccV = calcCCC(est_V, gt_V)
                cccA = calcCCC(est_A, gt_A)
                
                cccV2 = calcCCC(gt_V, gt_V)
                cccA2 = calcCCC(gt_A, gt_A)
                
                
                if lMSA > mseA : 
                    lMSA = mseA
                if lMSV > mseV : 
                    lMSV = mseV
                    
                if corA > lCRA : 
                    lCRA = corA
                if corV > lCRV : 
                    lCRV = corV
                    
                if cccA > lCCA : 
                    lCCA = cccA
                if cccV > lCCV : 
                    lCCV = cccV
                    
                if iccA > lICA : 
                    lICA = iccA
                if iccV > lICV : 
                    lICV = iccV
                    
                if (corA+corV+cccA+cccV+iccA+iccV) > total : 
                    total = (corA+corV+cccA+cccV+iccA+iccV)
                    if multi_gpu : 
                        torch.save(model_ft.module.state_dict(),curDir+'t-models/'+save_name+'-best')
                    else : 
                        torch.save(model_ft.state_dict(),curDir+'t-models/'+save_name+'-best')
                
                print('Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total))
                
                print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', CCCV2 : ',cccV2,', ICCV : ',iccV,', ICCV2 : ',iccV2)
                print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', CCCA2 : ',cccA2,', ICCA : ',iccA,', ICCA2 : ',iccA2)
                
                f = open(err_file,'a')
                res = 'MSEV : '+str(mseV)+ ', CORV : ' +str(corV)+', CCCV : '+str(cccV) +', ICCV : '+str(iccV)+' \n '
                f.write(res) 
                res = 'MSEA : '+str(mseA)+ ', CORA : '+str(corA) +', CCCA : '+str(cccA) +', ICCA : '+str(iccA)+' \n '
                f.write(res)
                res = 'Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total)+' \n '
                f.write(res)
                 
                f.close()

    print('Best val Acc: {:4f}'.format(lowest_loss))
    return 




def test_only_comb_seq():
    
    
    #64,0,1200 32,1,2000? 32,2,
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-split', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-addLoss', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-singleTask', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-trainQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-dConv', nargs='?', const=1, type=int, default=64)#64 
    parser.add_argument('-batch_size', nargs='?', const=1, type=int, default=6000) #0 is ori, 1 is red 

    parser.add_argument('-sewa', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-useWeightNormalization', nargs='?', const=1, type=int, default=1)#0,1,2
    
    parser.add_argument('-useAll', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-seq_length', nargs='?', const=1, type=int, default=4)#1,2,4,8,16,32
    
    parser.add_argument('-use_attention', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-use_ch', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-use_h', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-toLoad', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-toUpgrade', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-toAddAttention', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-per', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-numIters', nargs='?', const=1, type=int, default=200000)#0,1,2
    
    args = parser.parse_args()
    split = args.split
    addLoss = args.addLoss 
    singleTask = args.singleTask 
    isSewa = args.sewa 
    useWeight = args.useWeightNormalization
    useAll = args.useAll
    
    useAtt = args.use_attention
    useCH = args.use_ch
    useH = args.use_h
    
    trainQuadrant = args.trainQuadrant
    alterQuadrant = True
    
    per = args.per
    
    
    
    list_seq = [2,4,8,16,32]#[1]#[0,2,4,8,16,32]
    list_split = range(5)
    
    listRes = []
    
    c_dim=2
    image_size=128
    d_conv_dim=args.dConv
    inputC = 3#input channel for discriminator 
    isVideo = True
    toAlign = False
    
    toLoad = args.toLoad
    toUpgrade = args.toUpgrade
    toAddAttention = args.toAddAttention
    
    num_workers=1
    model_save_dir='stargan/models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    toRecordRes = True #use to get the metrics on the model's fodler. 
    toSave = False #use tosave to save the results to external folder
    
    
    dirTarget = "/media/deckyal/INT-450GB/extracted"
    
    for seq_length in (list_seq) : 
    
        root_dir = '/home/deckyal/Desktop/all-models/'
        sp_dir = '0-CH-SeqAfew-att-plus5'
        sq_dir = '/'+str(seq_length)+"/"
        
        theDir = root_dir+sp_dir+sq_dir
        resDir = theDir+'result/'
        checkDirMake(resDir)
        
        #seq_length = args.seq_length
        batch_size=int(truediv(args.batch_size,seq_length))#500, help='mini-batch size')
        
        
        evaluateSplit = True
        listRes = []
        if evaluateSplit  : 
        
            for split in list_split : 
                
                testSplit = split
                print("Test split " , testSplit)
                nSplit = 5
                listSplit = []
                for i in range(nSplit):
                    if i!=testSplit : 
                        listSplit.append(i)
                print(listSplit)
                
                if not isSewa : 
                    main_name = 'AF-C-'
                    d_name = 'AFEW-VA-Fixed'#'AFEW-VA-Fixed'
                    dbType = 0
                else : 
                    main_name = 'SE-C-'
                    d_name = 'SEWA'
                    dbType = 1
                
                if useH : 
                    main_name += 'R-'
                
                if useCH : 
                    main_name += 'CH-'
                    
                load_to_add = main_name
                
                if useAtt : 
                    main_name += 'A-'
                    
                load_to_add_split = main_name
                
                
                mseLoss = nn.MSELoss()
                
                main_name+=(str(d_conv_dim)+'-')
                load_to_add+=(str(d_conv_dim)+'-')
                load_to_add_split+=(str(d_conv_dim)+'-')
                
                if trainQuadrant : 
                    
                    if alterQuadrant : 
                        main_name+="-QDAL"
                        c_dim = 1
                    else :  
                        main_name+="-QD"
                        c_dim = 4
                
                save_name = main_name+str(testSplit)+'-n-'+str(seq_length)
                
                save_name_all = main_name+'all-'+str(seq_length)
                
                print('saving name is : ',save_name)
                VD = SEWAFEWReducedLatent([d_name], None, image_size, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
                            ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant,dbType = dbType,useAll = useAll,returnFName = toSave)
                
                dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)
               
                if not useH:
                    model_ft = CombinerSeqAtt(image_size, d_conv_dim, c_dim, 4,64,512,seq_length,batch_size,useCH=useCH)
                else : 
                    model_ft = CombinerSeqAttReplace(image_size, d_conv_dim, c_dim, 4,64,512,seq_length,batch_size,useCH=useCH)
                
                if toLoad:
                    print('loading previous model ')
                    model_ft.load_state_dict(torch.load(theDir+save_name))
                
                model_ft.to(device)
                model_ft.eval()
                
                listValO = []
                listAroO = []
                
                listValL = []
                listAroL = []
                
                tvo = [];tao=[];tvl = []; tal = [];
                anyDiffer = False
                
                print('not eval')
                #model_ft.eval()
                for x,(data) in enumerate(dataloaderV,0) :
                    
                    rinputs_l, rlabels_l,rldmrk_l = data[0],data[1],data[2]
                    
                    if toSave : 
                        fname_l = data[-1]
                    
                    model_ft.initialize(rinputs_l.shape[0])
                    
                    if useAtt : 
                        l_h = []
                        the_w = None
                    
                    with torch.set_grad_enabled(False) : 
                        
                        pre_result = None
                        
                        for y in range(seq_length): 
                            rinputs, rlabels, rldmrk = rinputs_l[:,y], rlabels_l[:,y],rldmrk_l[:,y]
                     
                            inputs = rinputs.cuda()#to(device) 
                            labels = rlabels.cuda()#to(device)
                            
                            if useAtt : 
                                if len(l_h) > 0:
                                    outputs,the_w = model_ft(inputs,prev_h = l_h,ret_w=True)
                                    print('w',the_w[:2])
                                else : 
                                    outputs = model_ft(inputs)
                                    
                                if useCH : 
                                    l_h.append(torch.cat((model_ft.lstm1_hdn[0][0],model_ft.lstm1_hdn[1][0]),1))
                                else : 
                                    l_h.append(model_ft.lstm1_hdn[0][0])
                            else :
                                outputs = model_ft(inputs) 
                                
                            #print('o shape',outputs.shape)
                            
                            print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                            
                            
                            
                            if outputs[:,0].shape[0] != batch_size : #in case the batch size is differ, usually at end of iter
                                anyDiffer = True 
                                print('differ')
                                tvo.append(outputs[:,0].detach().cpu())
                                tao.append(outputs[:,1].detach().cpu())
                                
                                tvl.append(labels[:,0].detach().cpu())
                                tal.append(labels[:,1].detach().cpu())
                            else :
                                print('equal')
                                listValO.append(outputs[:,0].detach().cpu())
                                listAroO.append(outputs[:,1].detach().cpu())
                                
                                listValL.append(labels[:,0].detach().cpu())
                                listAroL.append(labels[:,1].detach().cpu())
                                
                                
                            if toSave : 
                                if the_w is None : 
                                    the_w = labels.clone()
                                    the_w *=0
                                    
                                #print(fname_l)
                                #exit(0)
                                
                                for fn,pred,gt,tw in zip(fname_l[0],outputs.detach().cpu().numpy(),labels.detach().cpu().numpy(),the_w.detach().cpu().numpy()): 
                                    #print(fn,pred.shape,gt.shape,tw.shape)
                                    
                                    #1st get the file name 
                                    dirName, fName = os.path.split(fn)
                                    fName = fName.split('.')[0]
                                    #print('fname ',fName)
                                    
                                    print(fName,tw)
                                    
                                    listDir = dirName.split('/')
                                    
                                    indexName = listDir.index(d_name)
                                    folderName = os.path.join(dirTarget,d_name,listDir[indexName+1])
                                    
                                    folderNameImage = os.path.join(folderName,'img')
                                    folderNameRes = os.path.join(folderName,'resPred')
                                    folderNameW = os.path.join(folderName,'theW')
                                    
                                    checkDirMake(folderNameImage)
                                    checkDirMake(folderNameRes)
                                    checkDirMake(folderNameW)
                                    
                                    #original image path 
                                    listDir[-1] = 'img-128'
                                    imgPath = '/'.join(listDir)
                                     
                                    
                                    #check the image from actual gt, jpg etc. and save dummy file
                                    l_poss = ["jpg","jpeg",'png']
                                    imgName = None
                                    intr = 0
                                    imgName = imgPath+'/'+fName+"."+l_poss[intr]
                                    
                                    while (not os.path.isfile(imgName)): 
                                        #print('checking ',imgName)
                                        intr+=1
                                        imgName = imgPath+'/'+fName+"."+l_poss[intr]
                                        
                                    f = open(folderNameImage+'/'+fName+".txt",'w')
                                    f.write(imgName)
                                    f.close()
                                    #print('saved ',imgName,' to', folderNameImage+'/'+fName+".txt")
                                    
                                    
                                    #now save the pred,gt in npz
                                    np.savez(folderNameRes+'/'+fName+".npz",pred=pred,lbl=gt)
                                    
                                    #now save the tw in separate npz 
                                    np.save(folderNameW+'/'+fName+".npy",the_w)
                                    
                #exit(0)
                                    
                        
                if len(listValO) > 0 : 
                    est_V = np.asarray(torch.stack(listValO)).flatten()
                    est_A = np.asarray(torch.stack(listAroO)).flatten()
                    
                    gt_V = np.asarray(torch.stack(listValL)).flatten()
                    gt_A = np.asarray(torch.stack(listAroL)).flatten()
                
                if anyDiffer : 
                    est_Vt = np.asarray(torch.stack(tvo)).flatten()
                    est_At = np.asarray(torch.stack(tao)).flatten()
                    
                    gt_Vt = np.asarray(torch.stack(tvl)).flatten()
                    gt_At = np.asarray(torch.stack(tal)).flatten()
                    
                    #now concatenate
                    if len(listValO) > 0 : 
                        est_V = np.concatenate((est_V,est_Vt))
                        est_A = np.concatenate((est_A,est_At))
                        
                        gt_V = np.concatenate((gt_V,gt_Vt))
                        gt_A = np.concatenate((gt_A,gt_At))
                    else : 
                        est_V,est_A,gt_V,gt_A = est_Vt,est_At,gt_Vt,gt_At
                    
                print(est_V.shape, gt_V.shape)
                
                mseV = calcMSE(est_V, gt_V)
                mseA = calcMSE(est_A, gt_A)
                
                corV = calcCOR(est_V, gt_V)
                corA = calcCOR(est_A, gt_A)
                
                iccV = calcICC(est_V, gt_V)
                iccA = calcICC(est_A, gt_A)
                
                iccV2 = calcICC(gt_V, gt_V)
                iccA2 = calcICC(gt_A, gt_A)
                
                cccV = calcCCC(est_V, gt_V)
                cccA = calcCCC(est_A, gt_A)
                
                cccV2 = calcCCC(gt_V, gt_V)
                cccA2 = calcCCC(gt_A, gt_A)
                
                #print('Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total))
                
                print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', CCCV2 : ',cccV2,', ICCV : ',iccV,', ICCV2 : ',iccV2)
                print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', CCCA2 : ',cccA2,', ICCA : ',iccA,', ICCA2 : ',iccA2)
                
                res = np.asarray([[mseV,mseA],[corV,corA],[cccV,cccA],[iccV,iccA]])
                
                listRes.append(res)
                
                if toRecordRes : 
                    np.save(resDir+save_name+".npy",res)
                    print('saved : ',resDir+save_name+".npy")
                
                
                    with open(resDir+save_name+'.csv', 'w', newline='') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        
                        spamwriter.writerow([res[0,0],res[0,1]])
                        spamwriter.writerow([res[1,0],res[1,1]])
                        spamwriter.writerow([res[2,0],res[2,1]])
                        spamwriter.writerow([res[3,0],res[3,1]])
            
            if toRecordRes : 
                #now compiling the reesults from 5 split 
                listRes = np.stack(listRes)
                np.save(resDir+save_name_all+".npy",listRes)
                print('saved : ',resDir+save_name_all)
        
        if not isSewa : 
            main_name = 'AF-C-'
            d_name = 'AFEW-VA-Fixed'#'AFEW-VA-Fixed'
            dbType = 0
        else : 
            main_name = 'SE-C-'
            d_name = 'SEWA'
            dbType = 1
        
        if useH : 
            main_name += 'R-'
        
        if useCH : 
            main_name += 'CH-'
            
        load_to_add = main_name
        
        if useAtt : 
            main_name += 'A-'
            
        load_to_add_split = main_name
        
        mseLoss = nn.MSELoss()
        
        main_name+=(str(d_conv_dim)+'-')
        load_to_add+=(str(d_conv_dim)+'-')
        load_to_add_split+=(str(d_conv_dim)+'-')
        
        if trainQuadrant : 
            
            if alterQuadrant : 
                main_name+="-QDAL"
                c_dim = 1
            else :  
                main_name+="-QD"
                c_dim = 4
        
        
        save_name_all = main_name+'all-'+str(seq_length)
        listRes = np.load(resDir+save_name_all+".npy")
        print('loaded : ',resDir+save_name_all)
        print(listRes)
        
        
        l_m = []
        l_cor = []
        l_cc = []
        l_ic = []
        
        for tmp in range(listRes.shape[0]): 
            l_m.append(listRes[tmp][0,0]);l_m.append(listRes[tmp][0,1])
            l_cor.append(listRes[tmp][1,0]);l_cor.append(listRes[tmp][1,1])
            l_cc.append(listRes[tmp][2,0]);l_cc.append(listRes[tmp][2,1])
            l_ic.append(listRes[tmp][3,0]);l_ic.append(listRes[tmp][3,1])
            
        if toRecordRes: 
            with open(resDir+save_name_all+'.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                
                spamwriter.writerow(l_m)
                spamwriter.writerow(l_cor)
                spamwriter.writerow(l_cc)
                spamwriter.writerow(l_ic)
                
        print(np.stack(l_m))
        #now opening the file to make the csv 

if __name__ == '__main__':
     
    train_only_comb_seq() #To train seq C given extracted features of G and D
    test_only_comb_seq #To test the seq C
    