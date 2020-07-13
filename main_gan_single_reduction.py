import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from model import Generator, Discriminator, GeneratorM, GeneratorMZ, GeneratorMZR, DiscriminatorM,
DiscriminatorMST,DiscriminatorMZ,DiscriminatorMZR,DiscriminatorMZRL,CombinerSeqAtt 

from torch.autograd import Variable
from torchvision.utils import save_image
from FacialDataset import AFEWVA, AFEWVAReduced,SEWAFEWReduced
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
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('-split', nargs='?', const=1, type=int, default=0)#0,1,2
parser.add_argument('-sewa', nargs='?', const=1, type=int, default=0)#0,1,2
parser.add_argument('-semaine', nargs='?', const=1, type=int, default=1)#0,1,2

parser.add_argument('-gConv', nargs='?', const=1, type=int, default=16)#64
parser.add_argument('-dConv', nargs='?', const=1, type=int, default=16)#64 

parser.add_argument('-nSel', nargs='?', const=1, type=int, default=0) #0 is ori, 1 is red 
parser.add_argument('-batch_size', nargs='?', const=1, type=int, default=300) #0 is ori, 1 is red 
parser.add_argument('-multi_gpu', nargs='?', const=1, type=int, default=0) #0 is ori, 1 is red 
parser.add_argument('-resume_iters', nargs='?', const=1, type=int, default=79)#0,1,2. helpfull
parser.add_argument('-mode', nargs='?', const=1, type=int, default=0)#0 : train, 1 : extract

#may change
parser.add_argument('-tryDenoise', nargs='?', const=1, type=int, default=1)#0,1,2. Helpfull
parser.add_argument('-useWeightNormalization', nargs='?', const=0, type=int, default=1)#0,1,2. helpfull
parser.add_argument('-addLoss', nargs='?', const=1, type=int, default=1)#0,1,2. helpfull


#dont change 
parser.add_argument('-singleTask', nargs='?', const=1, type=int, default=0)#0,1,2. Multitask is slightly better
parser.add_argument('-trainQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2
parser.add_argument('-alterQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2
parser.add_argument('-useLatent', nargs='?', const=1, type=int, default=0)#0,1,2 #To use linear latent : bad
parser.add_argument('-useSkip', nargs='?', const=1, type=int, default=0)#0,1,2 #To use skip : no difference

args = parser.parse_args()

def str2bool(v):
    return v.lower() in ('true')
##############################################################



def toQuadrant(inputData = None, min = -10, max = 10,  toOneHot = False):
    threshold = truediv(min+max,2)
    vLow = False
    aLow = False
    q = 0
    
    #print(min,max)
    
    #print('the threshold : ',threshold)
    
    if inputData[0] < threshold : 
        vLow = True
    
    if inputData[1] < threshold : 
        aLow = True
    
    if vLow and aLow : 
        q = 2
    elif vLow and not aLow : 
        q = 1 
    elif not vLow and not aLow : 
        q = 0 
    else : 
        q = 3 
    
    if toOneHot : 
        rest = np.zeros(4)
        rest[q]+=1
        return rest 
    else : 
        return q 



def train_w_g_adl(): #training g and d on standard l2 loss
    
    split = args.split
    isSewa = args.sewa 
    isSemaine = args.semaine
     
    modelExist = True
    
    toLoadModel = True
    resume_iters=args.resume_iters#89
    
    GName = None;#"AF0-0-16-16-Den-UA-G-429.ckpt"
    DName = None;#"AF0-0-16-16-Den-UA-D-429.ckpt"
    
    use_skip = args.useSkip
    useLatent = args.useLatent
    tryDenoise = args.tryDenoise
    addLoss = args.addLoss
    useWeight = args.useWeightNormalization 
    
    singleTask = args.singleTask 
    trainQuadrant = args.trainQuadrant
    alterQuadrant = args.alterQuadrant
    
    nSel = args.nSel
    
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    image_size=128
    g_conv_dim=args.gConv
    d_conv_dim=args.dConv
    lambda_cls=1
    lambda_rec=10
    lambda_gp=10
    inputC = 3#input channel for discriminator 
    visEvery = 5
    saveEvery = 10    
    
    # Training configuration.
    dataset='CelebA' #, choices=['CelebA', 'RaFD', 'Both'])
    batch_size=args.batch_size#50#40#70#20 #, help='mini-batch size')
    num_iters=200000 #, help='number of total iterations for training D')
    num_iters_decay=100000 #, help='number of iterations for decaying lr')
    g_lr=0.0001 #, help='learning rate for G')
    d_lr=0.0001 #, help='learning rate for D')
    n_critic=5 #, help='number of D updates per each G update')
    beta1=0.5 #, help='beta1 for Adam optimizer')
    beta2=0.999 #, help='beta2 for Adam optimizer')
    #selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'] 
    #', '--list', nargs='+', help='selected attributes for the CelebA dataset',default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    
    isVideo = False
    seq_length = 2
    
    # Test configuration.
    test_iters=200000 #, help='test model from this step')

    # Miscellaneous.
    num_workers=1

    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples-g_adl'
    result_dir='stargan/results'

    # Step size.
    log_step=20
    sample_step=5#1000
    model_save_step=10
    lr_update_step=100#1000
    
    #model_save_step=10000
    #lr_update_step=1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    multi_gpu = args.multi_gpu
    testSplit = split
    print("Test split " , testSplit)
    nSplit = 5
    listSplit = []
    for i in range(nSplit):
        if i!=testSplit: 
            listSplit.append(i)
    print(listSplit)
    
    if isSemaine: 
        isSewa = 0
    
    if not isSewa : 
        if not isSemaine : 
            d_name = 'AFEW-VA-Fixed'
            additionName = "AF"+str(split)+"-"
        else : 
            d_name = 'Sem-Short'
            additionName = "SEM"+str(split)+"-"
        dbType = 0
    else : 
        d_name = 'SEWA'
        dbType = 1
        additionName = "SW"+str(split)+"-"
    
    additionName+=(str(nSel)+'-')
    additionName+=(str(g_conv_dim)+'-')
    additionName+=(str(d_conv_dim)+'-')
    
    if trainQuadrant : 
        if alterQuadrant : 
            additionName+="QDAL-"
            c_dim = 1
        else :  
            additionName+="QD-"
            c_dim = 4
            
    if tryDenoise :
        additionName+="Den-"
    
    
    transform =transforms.Compose([
            #transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    #AFEW-VA-Small
    
    ID = SEWAFEWReduced([d_name], None, True, image_size, transform, False, True, 1,split=True, nSplit = nSplit ,listSplit=listSplit
                ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType,
                returnWeight = useWeight,isSemaine = isSemaine)
    #ID = AFEWVA([d_name], None, True, image_size, transform, False, True, 1,split=True, nSplit = nSplit ,listSplit=listSplit
    #           ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType,returnWeight = useWeight)
    
    dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True,worker_init_fn=worker_init_fn)
    
    VD = SEWAFEWReduced([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
                ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType,
                isSemaine = isSemaine)
    #VD = AFEWVA([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
    #            ,isVideo=isVideo, seqLength = seq_length, returnNoisy = tryDenoise,dbType = dbType)
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)
   
    #Build model 
    """Create a generator and a discriminator."""
    
    if nSel :
        G = GeneratorMZ(g_conv_dim, 0, 1,use_skip,useLatent) 
        D = DiscriminatorMZR(image_size, d_conv_dim, c_dim, 4,inputC=inputC)
        C = CombinerSeqAtt(image_size, d_conv_dim, c_dim, 4,64,512,1,batch_size,useCH=True)
    else :
        G = GeneratorM(g_conv_dim, 0, 1,use_skip,useLatent)
        D = DiscriminatorM(image_size, d_conv_dim, c_dim, 6)
        C = CombinerSeqAtt(image_size, d_conv_dim, c_dim, 4,64,512,1,batch_size,useCH=True)
        
    
    print_network(G, 'G')
    print_network(D, 'D')
    
    if toLoadModel :
        print('Loading models from iterations : ',resume_iters) 
        
        if modelExist : 
            additionName+='UA-'
            
        if GName is None : 
            G_path = os.path.join(curDir+model_save_dir, '{}G-{}.ckpt'.format(additionName,resume_iters))
            D_path = os.path.join(curDir+model_save_dir, '{}D-{}.ckpt'.format(additionName,resume_iters))
            C_path = os.path.join(curDir+model_save_dir, '{}C-{}.ckpt'.format(additionName,resume_iters))
        else : 
            G_path = os.path.join(curDir+model_save_dir, GName)
            D_path = os.path.join(curDir+model_save_dir, DName)
            C_path = os.path.join(curDir+model_save_dir, DName)
        print('loading ',G_path)
        print('loading ',D_path)
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        C.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        
    if  not modelExist: 
        additionName+='UA-'
        
    else : 
        print('Initiating models')
        G.apply(weights_init_uniform_rule)
        D.apply(weights_init_uniform_rule)
        
    
    save_name = additionName+str(testSplit)
    err_file = curDir+save_name+".txt"
    
    
    print('err file : ',err_file)
    
    
    
        
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
    c_optimizer = torch.optim.Adam(C.parameters(), d_lr, [beta1, beta2])
    
    G.to(device)
    D.to(device)
    C.to(device)
    
    if multi_gpu: 
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        
    # Set data loader.
    data_loader = dataloader
    
    if not trainQuadrant or (alterQuadrant): 
        criterion = nn.MSELoss()
    else : 
        criterion = nn.CrossEntropyLoss() #F.cross_entropy(logit, target)

    # Fetch fixed inputs for debugging.
    data = next(iter(dataloader))
    x_fixed, rlabels,rldmrk,_ = data[0],data[1],data[2],data[3]#    x_fixed, c_org
    
    if trainQuadrant :
        if tryDenoise : 
            x_fixed = data[6].cuda()
            x_target = data[0].cuda()
    else : 
        if tryDenoise : 
            x_fixed = data[5].cuda()
            x_target = data[0].cuda()
        
    x_fixed = x_fixed.to(device)
    # Learning rate cache for decaying.
    d_lr = d_lr
    start_iters = 0
    # Start training.
    print('Start training...')
    start_time = time.time()
    
    if trainQuadrant : 
        q1 = data[4]
    
    f = open(err_file,'w+')
    f.write("err : ")
    f.close()
    
    lowest_loss = 99999
    
    lMSA,lMSV,lCCV,lCCA,lICA,lICV,lCRA, lCRV, total = 9999,9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999
    w,wv,wa = None,None,None
    
    for i in range(start_iters, num_iters):
        
        random.seed()
        manualSeed = random.randint(1, 10000) # use if you want new results
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        
        print('Epoch {}/{}'.format(i, num_iters - 1))
        print('-'*10)
        
        running_loss = 0
        
        G.train()
        D.train()
        
        for x,(data) in enumerate(dataloader,0) :
            rinputs, rlabels,rldmrk,_ =data[0],data[1],data[2],data[3]
            if trainQuadrant :
                if alterQuadrant : 
                    quadrant = data[5].float().cuda()
                else : 
                    quadrant = data[5].cuda()
                
                if tryDenoise : 
                    noisy = data[6].cuda()
                
            else : 
                if tryDenoise : 
                    noisy = data[5].cuda()
                
                    if useWeight : 
                        w = data[6].cuda()
                        #print(w)
                        wv = w[:,1]
                        wa = w[:,0]
                else : 
                    if useWeight : 
                        w = data[5].cuda()
                        #print(w)
                        wv = w[:,1]
                        wa = w[:,0]
            
            inputs = rinputs.cuda()#to(device)
            labels = rlabels.cuda()#to(device)
            
            # Compute loss with real images.
            out_src, out_cls = D(inputs)
            d_loss_real = - torch.mean(out_src)
            
            if not trainQuadrant: 
                if useWeight :
                    d_loss_cls = calcMSET(out_cls,labels,w) #criterion(out_cls, labels)
                else : 
                    d_loss_cls = criterion(out_cls, labels) #classification_loss(out_cls, label_org, dataset)
                    
                if addLoss :
                    ov,oa,lv,la = out_cls[:,0],out_cls[:,1], labels[:,0], labels[:,1]
                    
                    corV = -calcCORT(ov, lv, wv)
                    corA = -calcCORT(oa, la, wa)
                    
                    cccV = -calcCCCT(ov, lv, wv)
                    cccA = -calcCCCT(oa, la, wa)
                    
                    iccV = -calcICCT(ov, lv, wv)
                    iccA = -calcICCT(oa, la, wa)
                    
                    d_loss_cls = d_loss_cls + corV+corA +cccV+cccA+iccV+iccA
            else :
                #print('q ',quadrant)
                #print(out_cls.shape, quadrant.shape )
                if alterQuadrant : 
                    d_loss_cls = criterion(torch.squeeze(out_cls), quadrant)
                else : 
                    d_loss_cls = criterion(out_cls, quadrant)
            
            if x%10 == 0 : 
                if not trainQuadrant: 
                    print(x,'-',len(dataloader)," Res - label-G : ", out_cls[:3],labels[:3])
                else : 
                    if alterQuadrant :
                        print(x,'-',len(dataloader)," Res - label-G : ", torch.round(out_cls[:3]),quadrant[:3]) 
                    else : 
                        print(x,'-',len(dataloader)," Res - label-G : ", torch.max(out_cls[:3],1)[1],quadrant[:3])
            
            # Compute loss with fake images.
            
            
            if tryDenoise :
                theInput = noisy 
            else : 
                theInput = inputs 
                
            x_fake = G(theInput)
            
            out_src, out_cls = D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)
    
            # Compute loss for gradient penalty.
            alpha = torch.rand(theInput.size(0), 1, 1, 1).to(device)
            x_hat = (alpha * theInput.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            
            out_src, _ = D(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat)
    
            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls + lambda_gp * d_loss_gp
            
            
            #reset_grad()
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            d_loss.backward()
            d_optimizer.step()
            
            
            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            
            ###! Actual training of the generator 
                    
            if (i+1) % n_critic == 0:
                
                # Original-to-target domain.
                if tryDenoise : 
                    z,x_fake = G(noisy,returnInter = True)
                else : 
                    z,x_fake = G(inputs)
                out_src, out_cls = D(x_fake)
                
                if x%10 == 0 : 
                    print("Res - label-D : ", out_cls[:3],labels[:3])
                    
                g_loss_fake = - torch.mean(out_src)
                
                if not trainQuadrant: 
                    #g_loss_cls = criterion(out_cls, labels) #classification_loss(out_cls, label_org, dataset)
                    
                    if useWeight :
                        g_loss_cls = calcMSET(out_cls,labels,w) #criterion(out_cls, labels)
                    else : 
                        g_loss_cls = criterion(out_cls, labels) #classification_loss(out_cls, label_org, dataset)
                        
                    if addLoss :
                        ov,oa,lv,la = out_cls[:,0],out_cls[:,1], labels[:,0], labels[:,1]
                        
                        corV = -calcCORT(ov, lv, wv)
                        corA = -calcCORT(oa, la, wa)
                        
                        cccV = -calcCCCT(lv, lv, wv)
                        cccA = -calcCCCT(oa, la, wa)
                        
                        iccV = -calcICCT(ov, lv, wv)
                        iccA = -calcICCT(oa, la, wa)
                        
                        g_loss_cls = g_loss_cls + corV+corA +cccV+cccA+iccV+iccA
                        
                    
                else : 
                    if alterQuadrant : 
                        g_loss_cls = criterion(torch.squeeze(out_cls), quadrant)
                    else : 
                        g_loss_cls = criterion(out_cls, quadrant)
    
                
                if not isSewa: 
                    q = toQuadrant(out_cls, -10, 10, False)
                else : 
                    q = toQuadrant(out_cls, 0, 1, False)
                
                out_c = C(torch.cat((z,q),1))
                if useWeight :
                    c_loss = calcMSET(out_cls,labels,w) #criterion(out_cls, labels)
                else : 
                    c_loss = criterion(out_cls, labels) #classification_loss(out_cls, label_org, dataset)
                    
                
                if addLoss :
                    ov,oa,lv,la = out_c[:,0],out_c[:,1], labels[:,0], labels[:,1]
                    
                    corV = -calcCORT(ov, lv, wv)
                    corA = -calcCORT(oa, la, wa)
                    
                    cccV = -calcCCCT(lv, lv, wv)
                    cccA = -calcCCCT(oa, la, wa)
                    
                    iccV = -calcICCT(ov, lv, wv)
                    iccA = -calcICCT(oa, la, wa)
                    
                    c_loss = c_loss + corV+corA +cccV+cccA+iccV+iccA
                
                
                
                
                # Target-to-original domain.
                x_reconst = G(x_fake)
                g_loss_rec = torch.mean(torch.abs(inputs - x_reconst))
    
                # Backward and optimize.
                g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls
                
                #reset_grad()    
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                c_optimizer.zero_grad()
                
                c_loss.backward()
                
                g_loss.backward()
                g_optimizer.step()
                c_optimizer.step()
    
    
                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['C'] = c_loss.item()
                
                ###! Getting the training metrics and samples    
                #running_loss += loss.item() * inputs.size(0)
                #print("{}/{} loss : {}/{}".format(x,int(len(dataloader.dataset)/batch_size),lossC.item(),lossR.item()))
                
             
            if (i+1) % 10 == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}], Inner {}/{} \n".format(et, i+1, num_iters,x,int(len(dataloader.dataset)/batch_size))
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                
                
                f = open(err_file,'a')
                f.write("Elapsed [{}], Iteration [{}/{}], Inner {}/{} \n".format(et, i+1, num_iters,x,int(len(dataloader.dataset)/batch_size)))
                f.write(log) 
                f.close()
                    
        # Translate fixed images for debugging.
        if (i+1) % visEvery == 0:
            with torch.no_grad():
                x_fake_list = [x_fixed]
                x_concat = G(x_fixed)
                sample_path = os.path.join(curDir+sample_dir, '{}{}-images-denoised.jpg'.format(i+1,additionName))
                save_image(denorm(x_concat.data.cpu()), sample_path, nrow=int(round(batch_size/4)), padding=0)
                print('Saved real and fake denoised images into {}...'.format(sample_path))
                
                if tryDenoise : 
                    x_concat = x_fixed
                    sample_path = os.path.join(curDir+sample_dir, '{}{}-images-original.jpg'.format(i+1,additionName))
                    save_image(denorm(x_concat.data.cpu()), sample_path, nrow=int(round(batch_size/4)), padding=0)
                    print('Saved real and fake real images into {}...'.format(sample_path))
                    
                    x_concat = x_target
                    sample_path = os.path.join(curDir+sample_dir, '{}{}-images-groundtruth.jpg'.format(i+1,additionName))
                    save_image(denorm(x_concat.data.cpu()), sample_path, nrow=int(round(batch_size/4)), padding=0)
                    print('Saved real and fake real images into {}...'.format(sample_path))

        # Save model checkpoints.
        if (i+1) % saveEvery == 0:
            G_path = os.path.join(curDir+model_save_dir, '{}G-{}.ckpt'.format(additionName,i))
            D_path = os.path.join(curDir+model_save_dir, '{}D-{}.ckpt'.format(additionName,i))
            C_path = os.path.join(curDir+model_save_dir, '{}C-{}.ckpt'.format(additionName,i))
            
            if multi_gpu : 
                torch.save(G.module.state_dict(), G_path)
                torch.save(D.module.state_dict(), D_path)
                torch.save(C.module.state_dict(), C_path)
            else : 
                torch.save(G.state_dict(), G_path)
                torch.save(D.state_dict(), D_path)
                torch.save(C.state_dict(), C_path)
            
            print('Saved model checkpoints into {}...'.format(model_save_dir))
            print(G_path)

        # Decay learning rates.
        if (i+1) % lr_update_step == 0 and (i+1) > 50:
            g_lr -= (g_lr / float(num_iters_decay))
            d_lr -= (d_lr / float(num_iters_decay))
            update_lr_ind(d_optimizer,d_lr)
            update_lr_ind(g_optimizer,g_lr)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print('Loss : {:.4f}'.format(epoch_loss))
        
        if i %2 == 0 : 
            if multi_gpu : 
                torch.save(D.module.state_dict(),curDir+'t-models/'+'-D'+save_name)
                torch.save(G.module.state_dict(),curDir+'t-models/'+'-G'+save_name)
                torch.save(C.module.state_dict(),curDir+'t-models/'+'-C'+save_name)
            else : 
                torch.save(D.state_dict(),curDir+'t-models/'+'-D'+save_name)
                torch.save(G.state_dict(),curDir+'t-models/'+'-G'+save_name)
                torch.save(G.state_dict(),curDir+'t-models/'+'-C'+save_name)
    
        #Deep copy the model_ft 
        if i%5 == 0 :#epoch_loss < lowest_loss : 
            
            
            if trainQuadrant : 
                a = 0
                b = 0
            else : 
                a = 0
                b = 1
            
            lowest_loss = lowest_loss 
            
            print("outp8ut : ",out_cls[0])
            print("labels : ",labels[0])
            
            if True : 
            
                listValO = []
                listAroO = []
                
                listValL = []
                listAroL = []
                
                tvo = [];tao=[];tvl = []; tal = [];
                anyDiffer = False
               
                for x,(data) in enumerate(dataloaderV,0) :
                    
                    if trainQuadrant: 
                        rinputs, rlabels,rldmrk = data[0],data[5],data[2]
                    else : 
                        rinputs, rlabels,rldmrk = data[0],data[1],data[2]
                    
                    G.eval()    
                    D.eval()
                    C.eval()
                    
                    inputs = rinputs.cuda()#to(device) 
                    labels = rlabels.cuda()#to(device)
                    
                    with torch.set_grad_enabled(False) : 
                        
                        z,inputsM = G(inputs,returnInter = True)
                        _, outD = D(inputsM)
                        
                        if not isSewa: 
                            q = toQuadrant(outD, -10, 10, False)
                        else : 
                            q = toQuadrant(outD, 0, 1, False)
                        
                        outputs = C(torch.cat((z,q),1))
                        
                        if trainQuadrant: 
                            if alterQuadrant :
                                outputs = torch.round(outputs) 
                            else : 
                                _,outputs = torch.max(outputs,1)
                        
                        if trainQuadrant : 
                            print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs.shape)
                        else : 
                            print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                        #print(outputs.shape)
                        
                        if not trainQuadrant : 
                            shape = outputs[:,0].shape[0]
                        else : 
                            shape = outputs.shape[0]
                        
                        if shape != batch_size : #in case the batch size is differ, usually at end of iter
                            anyDiffer = True 
                            print('differ')
                            if trainQuadrant: 
                                tvo.append(outputs.detach().cpu())
                                tao.append(outputs.detach().cpu())
                                
                                tvl.append(labels.detach().cpu())
                                tal.append(labels.detach().cpu())
                            else : 
                                tvo.append(outputs[:,a].detach().cpu())
                                tao.append(outputs[:,b].detach().cpu())
                                
                                tvl.append(labels[:,a].detach().cpu())
                                tal.append(labels[:,b].detach().cpu())
                        else :
                            print('equal')
                            if trainQuadrant : 
                                listValO.append(outputs.detach().cpu())
                                listAroO.append(outputs.detach().cpu())
                                
                                listValL.append(labels.detach().cpu())
                                listAroL.append(labels.detach().cpu())
                            else : 
                                listValO.append(outputs[:,a].detach().cpu())
                                listAroO.append(outputs[:,b].detach().cpu())
                                
                                listValL.append(labels[:,a].detach().cpu())
                                listAroL.append(labels[:,b].detach().cpu())
                                
                        
                
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
                    est_V = np.concatenate((est_V,est_Vt))
                    est_A = np.concatenate((est_A,est_At))
                    
                    gt_V = np.concatenate((gt_V,gt_Vt))
                    gt_A = np.concatenate((gt_A,gt_At))
                    
                print(est_V.shape, gt_V.shape)
                
                mseV = calcMSE(est_V, gt_V)
                mseA = calcMSE(est_A, gt_A)
                
                corV = calcCOR(est_V, gt_V)
                corA = calcCOR(est_A, gt_A)
                
                iccV = calcICC(est_V, gt_V)
                iccA = calcICC(est_A, gt_A)
                
                cccV = calcCCC(est_V, gt_V)
                cccA = calcCCC(est_A, gt_A)
                
                iccV2 = calcCCC(gt_V, gt_V)
                iccA2 = calcCCC(gt_A, gt_A)
                
                
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
                    G_path = os.path.join(curDir+model_save_dir, '{}G-best-{}.ckpt'.format(additionName,i))
                    D_path = os.path.join(curDir+model_save_dir, '{}D-best-{}.ckpt'.format(additionName,i))
                    
                    #G_path = os.path.join(curDir+model_save_dir, '{}{}-G-adl-best.ckpt'.format(i+1,additionName))
                    #D_path = os.path.join(curDir+model_save_dir, '{}{}-D-adl-best.ckpt'.format(i+1,additionName))
                    
                    if multi_gpu :
                        torch.save(G.module.state_dict(), G_path)
                        torch.save(D.module.state_dict(), D_path)
                    else : 
                        torch.save(G.state_dict(), G_path)
                        torch.save(D.state_dict(), D_path)
                
                
                print('Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total))
                
                print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', ICCV : ',iccV)
                print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', ICCA : ',iccA)
                
                f = open(err_file,'a')
                res = 'MSEV : '+str(mseV)+ ', CORV : ' +str(corV)+', CCCV : '+str(cccV) +', ICCV : '+str(iccV)+' \n '
                f.write(res) 
                res = 'MSEA : '+str(mseA)+ ', CORA : '+str(corA) +', CCCA : '+str(cccA) +', ICCA : '+str(iccA)+' \n '
                f.write(res)
                res = 'Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total)+' \n '
                f.write(res)
                 
                f.close()

    print('Best val Acc: {:4f}'.format(lowest_loss)) 
    pass

def extract(): #training g and d on standard l2 loss
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    split = args.split
    isSewa = args.sewa 
    isSemaine = args.semaine 
    
    
    toLoadModel = True
    resume_iters=args.resume_iters
    
    use_skip = args.useSkip
    useLatent = args.useLatent
    tryDenoise = args.tryDenoise
    addLoss = args.addLoss
    useWeight = args.useWeightNormalization 
    
    singleTask = args.singleTask 
    trainQuadrant = args.trainQuadrant
    alterQuadrant = args.alterQuadrant
    
    nSel = args.nSel
    
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    image_size=128
    g_conv_dim=16
    d_conv_dim=16
    lambda_cls=1
    lambda_rec=10
    lambda_gp=10
    inputC = 3#input channel for discriminator 
    
   
    batch_size=args.batch_size#200 #50#40#70#20 #, help='mini-batch size')
   
    isVideo = False
    seq_length = 2
    
    # Test configuration.
    test_iters=200000 #, help='test model from this step')

    # Miscellaneous.
    num_workers=1

    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples-g_adl'
    result_dir='stargan/results'

    # Step size.
    log_step=20
    sample_step=5#1000
    model_save_step=10
    lr_update_step=100#1000
    
    #model_save_step=10000
    #lr_update_step=1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
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
        if not isSemaine : 
            d_name = 'AFEW-VA-Fixed'
            additionName = "AF"+str(split)+"-"
        else : 
            d_name = 'Sem-Short'
            additionName = "SEM"+str(split)+"-"
        dbType = 0
    else : 
        d_name = 'SEWA'
        dbType = 1
        additionName = "SW"+str(split)+"-"
    
    
    additionName+=(str(nSel)+'-')
    additionName+=(str(g_conv_dim)+'-')
    additionName+=(str(d_conv_dim)+'-')
    
    if trainQuadrant : 
        if alterQuadrant : 
            additionName+="QDAL-"
            c_dim = 1
        else :  
            additionName+="QD-"
            c_dim = 4
            
    if tryDenoise :
        additionName+="Den-"
        
    save_name = additionName+str(testSplit)
    
    transform =transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    toDelete = False
    
    
    VD = SEWAFEWReduced([d_name], None, True, image_size, transform, False, False, 1,split=False, nSplit = nSplit,listSplit=[testSplit]
                ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType, isSemaine=isSemaine)
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)
   
    if nSel :
        G = GeneratorMZ(g_conv_dim, 0, 1,use_skip,useLatent) 
        D = DiscriminatorMZR(image_size, d_conv_dim, c_dim, 4,inputC=inputC)
    else :
        G = GeneratorM(g_conv_dim, 0, 1,use_skip,useLatent)
        D = DiscriminatorM(image_size, d_conv_dim, c_dim, 6)
        
    
    print_network(G, 'G')
    print_network(D, 'D')
    
    if toLoadModel :
        print('Loading models from iterations : ',resume_iters) 
        G_path = os.path.join(curDir+model_save_dir, '{}G-{}.ckpt'.format(additionName,resume_iters))
        D_path = os.path.join(curDir+model_save_dir, '{}D-{}.ckpt'.format(additionName,resume_iters))
        
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage),strict=False)
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage),strict=False)
        
    G.to(device)
    D.to(device)
    
    
    
    
    listValO = []
    listAroO = []
    
    listValL = []
    listAroL = []

    a = 0
    b = 1
    
    iterator = 0
    
    
    tvo = [];tao=[];tvl = []; tal = [];
    anyDiffer = False
    
    print('length : ',len(dataloaderV))
    
    for x,(data) in enumerate(dataloaderV,0) :
        
        if trainQuadrant: 
            rinputs, rlabels,rldmrk = data[0],data[5],data[2]
        else : 
            rinputs, rlabels,rldmrk = data[0],data[1],data[2]
        
        #for real_batch,va,gt,M,ln,q,noisy_batch,weight in (dataloader) :
        fNames = data[4]
        
        G.train()    
        D.train()
        
        inputs = rinputs.cuda()#to(device) 
        labels = rlabels.cuda()#to(device)
        
        with torch.set_grad_enabled(False) : 
            
            inputsM,z = G(inputs,returnInter = True)
            _, outputs = D(inputsM)
            
            if trainQuadrant: 
                if alterQuadrant :
                    outputs = torch.round(outputs) 
                else : 
                    _,outputs = torch.max(outputs,1)
            
            print('inside ')
            
            if trainQuadrant : 
                print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs.shape)
            else : 
                print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
            #print(outputs.shape)
            
            print(z.shape)
            zSave = z.cpu().numpy()
            qSave = outputs.cpu().numpy()
            
            combine = True
            
            #now saving the results individually 
            for fname,features,va in zip(fNames, zSave,qSave):
                iterator+=1
                
                #first inspect the dir 
                dirName, fName = os.path.split(fname)
                
                fName = fName.split('.')[0]+'.npz'
                    
                listDir = dirName.split('/')
                listDir[-1] = 'FT-'+additionName+'z'
                
                dirTgt = '/'.join(listDir)
                
                if not toDelete : 
                    checkDirMake(dirTgt)
                
                #va = np.array([5,-5])
                
                #print(va)
                if not isSewa: 
                    q = toQuadrant(va, -10, 10, False)
                else : 
                    q = toQuadrant(va, 0, 1, False)
                #print(q)
                
                
                if combine : 
                    tmp=np.zeros((1,features.shape[1],features.shape[2]),np.float32)+q
                    features=np.concatenate((features,tmp),0)
                    print(tmp[0,0,:2])
                
                print(fname, features.shape) 
                
                if os.path.isdir(dirTgt) and toDelete: # and isSewa or False: 
                    print('removing : ',dirTgt)
                    #os.remove(os.path.join(dirTgt,fNameOri))
                    #exit(0)
                    shutil.rmtree(dirTgt)
                #print(dirTgt, fName)
                
                vaq = np.array([va[0],va[1],q])
                #print('vaq : ',vaq)
                
                if not toDelete :#not os.path.isfile(os.path.join(dirTgt,fName)) : 
                    #np.save(os.path.join(dirTgt,fName),features)
                    np.savez(os.path.join(dirTgt,fName),z=features,vaq=vaq)
                    
                #exit(0)
                                      
            #np.save('testing.npy',zSave)
            #exit(0)
            
            if not trainQuadrant : 
                shape = outputs[:,0].shape[0]
            else : 
                shape = outputs.shape[0]
            
            if shape != batch_size : #in case the batch size is differ, usually at end of iter
                anyDiffer = True 
                print('differ')
                if trainQuadrant: 
                    tvo.append(outputs.detach().cpu())
                    tao.append(outputs.detach().cpu())
                    
                    tvl.append(labels.detach().cpu())
                    tal.append(labels.detach().cpu())
                else : 
                    tvo.append(outputs[:,a].detach().cpu())
                    tao.append(outputs[:,b].detach().cpu())
                    
                    tvl.append(labels[:,a].detach().cpu())
                    tal.append(labels[:,b].detach().cpu())
            else :
                print('equal')
                if trainQuadrant : 
                    listValO.append(outputs.detach().cpu())
                    listAroO.append(outputs.detach().cpu())
                    
                    listValL.append(labels.detach().cpu())
                    listAroL.append(labels.detach().cpu())
                else : 
                    listValO.append(outputs[:,a].detach().cpu())
                    listAroO.append(outputs[:,b].detach().cpu())
                    
                    listValL.append(labels[:,a].detach().cpu())
                    listAroL.append(labels[:,b].detach().cpu())
                    
            
    
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
    
    cccV = calcCCC(est_V, gt_V)
    cccA = calcCCC(est_A, gt_A)
    
    iccV2 = calcCCC(gt_V, gt_V)
    iccA2 = calcCCC(gt_A, gt_A)
    
    
    print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', ICCV : ',iccV)
    print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', ICCA : ',iccA)

if __name__ == '__main__':
    mode = args.mode
    if mode == 0 : #To train GDC
        train_w_g_adl()
    elif mode == 1 : #To extract the features
        extract()
    
    
