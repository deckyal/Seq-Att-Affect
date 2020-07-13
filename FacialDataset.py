from math import sqrt
import re

from PIL import Image,ImageFilter

import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import csv
import torchvision.transforms.functional as F
import numbers
from torchvision.transforms import RandomRotation,RandomResizedCrop,RandomHorizontalFlip



from utils import *
from config import *
from ImageAugment import *
import utils
from os.path import isfile# load additional module
import pickle
import os
#import nudged
import shutil
import file_walker
import copy
from random import randint

#noiseParamList = np.asarray([[0,0,0],[1,2,3],[1,3,5],[.001,.005,.01],[.8,.5,.2],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
noiseParamList =np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]

#noiseParamListTrain = np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
noiseParamListTrain = np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]

rootDir = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data"
rootDirLdmrk = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"

def addGaussianNoise(img,noiseLevel = 1):
    noise = torch.randn(img.size()) * noiseLevel
    noisy_img = img + noise
    return noisy_img



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
    
    

class SEWAFEW(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AFEW"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,split = False, 
                 nSplit = 5, listSplit = [0,1,2,3,4],wHeatmap= False,isVideo = False, seqLength = None,
                 returnM = False, toAlign = False, dbType = 0):#dbtype 0 is AFEW, 1 is SEWA
        
        self.dbType = dbType
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        self.align = toAlign
        self.useNudget = False
        self.returnM = returnM
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        self.wHeatmap = wHeatmap
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDir+"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
        
        if self.dbType ==1 : 
            annotL_name = "annotOri"
            self.ldmrkNumber = 49
            self.nose = 16
            self.leye = 24
            self.reye = 29
            #mean_shape49-pad3-224
            
            self.mean_shape = np.load(curDir+'mean_shape49-pad3-'+str(image_size)+'.npy')
        else :
            annotL_name = 'annot'
            self.ldmrkNumber = 68
            self.nose = 33
            self.leye = 41
            self.reye = 46
            
            self.mean_shape = np.load(curDir+'mean_shape-pad-'+str(image_size)+'.npy')
        
        self.swap = False 
        
        if self.swap : 
            self.ptsDst = np.asarray([
                [self.mean_shape[self.nose+self.ldmrkNumber],self.mean_shape[self.nose]],[self.mean_shape[self.leye+self.ldmrkNumber],self.mean_shape[self.leye]],[self.mean_shape[self.reye+self.ldmrkNumber],self.mean_shape[self.reye]]
                ],dtype= np.float32)
            
            self.ptsTn = [self.mean_shape[self.nose+self.ldmrkNumber],self.mean_shape[self.nose]],[self.mean_shape[self.leye+self.ldmrkNumber],self.mean_shape[self.leye]],[self.mean_shape[self.reye+self.ldmrkNumber],self.mean_shape[self.reye]]
        else : 
            self.ptsDst = np.asarray([
                [self.mean_shape[self.nose],self.mean_shape[self.nose+self.ldmrkNumber]],[self.mean_shape[self.leye],self.mean_shape[self.leye+self.ldmrkNumber]],[self.mean_shape[self.reye],self.mean_shape[self.reye+self.ldmrkNumber]]
                ],dtype= np.float32)
            self.ptsTn = [self.mean_shape[self.nose],self.mean_shape[self.nose+self.ldmrkNumber]],[self.mean_shape[self.leye],self.mean_shape[self.leye+self.ldmrkNumber]],[self.mean_shape[self.reye],self.mean_shape[self.reye+self.ldmrkNumber]]
            
            self.ptsTnFull = np.column_stack((self.mean_shape[:self.ldmrkNumber],self.mean_shape[self.ldmrkNumber:]))
        
        list_gt = []
        list_labels_t = []
        list_labels_tE = []
        
        counter_image = 0
        
        annotE_name = 'annot2'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        list_missing = []
        
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(self.curDir +data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    #c_image,c_ldmark = 0,0
                    
                    if self.dbType == 1 : #we directly get the VA file in case of sewa  
                        #first get the valence 
                        valFile = f.full_path+"/valence/"+f.name+"_Valence_A_Aligned.csv"
                        aroFile = f.full_path+"/arousal/"+f.name+"_Arousal_A_Aligned.csv"
                        
                        list_labels_tE.append([valFile,aroFile])
                        #print(valFile,aroFile)
                        
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            
                            #print(sub_f.name)
                            if(sub_f.name == annotL_name) : #If that's annot, add to labels_t
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_t.append(sorted(list_dta))
                                c_image = len(list_dta)
                            elif(sub_f.name == 'img'): #Else it is the image
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
                                c_ldmrk = len(list_dta)
        
                            elif (sub_f.name == annotE_name) :
                                if self.dbType == 0 : 
                                    #If that's annot, add to labels_t
                                    for sub_sub_f in sub_f.walk(): #this is the data
                                        if(".npy" not in sub_sub_f.full_path):
                                            list_dta.append(sub_sub_f.full_path)
                                    list_labels_tE.append(sorted(list_dta))
                                    
                    if(c_image!=c_ldmrk) and False: 
                        print(f.full_path," is incomplete ",'*'*10,c_image,'-',c_ldmrk)
                        ori = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/allVideo/"
                        target = '/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/allVideo/retrack/'
                        #shutil.copy(ori+f.name+".avi",target+f.name+".avi")
                        list_missing.append(f.name)
                                    
        self.length = counter_image
        print("Now opening keylabels")
        
        list_labelsN = [] 
        list_labelsEN = []
        
        list_labels = [] 
        list_labelsE = []
        
        for ix in range(len(list_labels_t)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            lbl_68 = [] #Per folder
            lbl_2 = [] #Per folder
            
            lbl_n68 = [] #Per folder
            lbl_n2 = [] #Per folder
            for jx in range(len (list_labels_t[ix])): #lbl_sub in lbl :
                
                #print(os.path.basename(list_gt[ix][jx]))
                #print(os.path.basename(list_labels_t[ix][jx]))
                #print(os.path.basename(list_labels_tE[ix][jx]))
                
                lbl_sub = list_labels_t[ix][jx]
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    lbl_68.append(read_kp_file(lbl_sub,True))
                        
                    lbl_n68.append(lbl_sub)
                
                if self.dbType == 0 : 
                    lbl_subE = list_labels_tE[ix][jx]
                    if ('aro' in lbl_subE) : 
                        x = []
                        #print(lbl_sub)
                        with open(lbl_subE) as file:
                            data2 = [re.split(r'\t+',l.strip()) for l in file]
                        for i in range(len(data2)) :
                            #x.append([ float(j) for j in data2[i][0].split()] )
                            temp = [ float(j) for j in data2[i][0].split()]
                            temp.reverse() #to give the valence first. then arousal
                            x.append(temp)
                        
                        #x.reverse()
                        
                        lbl_2.append(np.array(x).flatten('F'))
                        lbl_n2.append(lbl_sub)
                
            if self.dbType == 1 : #sewa  
                #print(list_labels_t[ix][0])
                valFile = np.asarray(readCSV(list_labels_tE[ix][0]))
                aroFile = np.asarray(readCSV(list_labels_tE[ix][1]))
                
                lbl_n2.append(list_labels_tE[ix][0])
                lbl_2 = np.column_stack((valFile,aroFile))
                
            
            list_labelsN.append(lbl_n68)
            list_labelsEN.append(lbl_n2)
            
            list_labels.append(lbl_68)
            list_labelsE.append(lbl_2)
                
        
        t_l_imgs = []
        t_l_gt = []
        t_l_gtE = []
        
        t_list_gt_names = []
        t_list_gtE_names = []
        
        #print(list_labelsEN)
        
        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gt_names = []
                list_gtE_names = []
                indexer = 0
                
                list_ground_truth = np.zeros([len(list_gt[i]),self.ldmrkNumber*2])
                list_ground_truthE = np.zeros([len(list_gt[i]),2])
                
                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])
                    
                    list_gt_names.append(list_labelsN[i][j])
                    if self.dbType == 0 : 
                        list_gtE_names.append(list_labelsEN[i][j])
                    else : 
                        list_gtE_names.append(list_labelsEN[i][0])
                    #print(list_labelsEN[i])
                    
                    '''if len(list_labels[i][j] < 1): 
                        print(list_labels[i][j])'''
                    #print(len(list_labels[i][j]))
                    list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_imgs.append(list_images)
                t_l_gt.append(list_ground_truth)
                t_l_gtE.append(list_ground_truthE)
                
                t_list_gt_names.append(list_gt_names)
                t_list_gtE_names.append(list_gtE_names)
        
        else : 
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;    
                    list_gt_names = []
                    list_gtE_names = []
                    
                    list_ground_truth = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,136]) #np.zeros([counter_image,136])
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize   
                        tmpn68 = []
                        tmpn2 = []
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z].flatten('F')
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                             
                            tmpn68.append(list_labelsN[i][z])
                            tmpn2.append(list_labelsEN[i][z])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                        list_ground_truthE[indexer] = temp3
                        
                        list_gt_names.append(tmpn68)
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gt.append(list_ground_truth)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gt_names.append(list_gt_names)
                    t_list_gtE_names.append(list_gtE_names)
                
        self.l_imgs = []
        self.l_gt = []
        self.l_gtE = []
        
        self.list_gt_names = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if split :
            indexer = 0 
            
            self.l_gt = []
            self.l_gtE = []
                 
            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                
                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): 
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gt.append(t_l_gt[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])
                            
                            self.list_gt_names.append(t_list_gt_names[i][j])
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1
                            
                else : 
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter
                        
                            t_img = []
                            t_gt = []
                            t_gtE = []
                            t_gt_N = []
                            t_gt_EN = []
                            tmp = 0
                            
                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gt.append(t_l_gt[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])
                                
                                t_gt_N.append(t_list_gt_names[i][j][k])
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1
                                
                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gt.append(t_gt)
                            self.l_gtE.append(t_gtE)
                            
                            self.list_gt_names.append(t_gt_N)
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1
                    
                print(len(self.l_imgs))
                    
            self.l_gt = np.asarray(self.l_gt)
            self.l_gtE = np.asarray(self.l_gtE)
        else :
            if not self.isVideo :
                self.l_gt = np.zeros([counter_image,136])
                self.l_gtE = np.zeros([counter_image,2])
                indexer = 0
                
                
                for i in range(len(t_l_imgs)): 
                    for j in range(len(t_l_imgs[i])): 
                        self.l_imgs.append(t_l_imgs[i][j])
                        print(i,j,'-',len(t_l_imgs[i]))
                        self.l_gt[indexer] = t_l_gt[i][j]
                        self.l_gtE[indexer] = t_l_gtE[i][j]
                        
                        self.list_gt_names.append(t_list_gt_names[i][j])
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1
                    
            else : 
                self.l_gt= np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])
                
                indexer = 0
                
                for i in range(len(t_l_imgs)): #dataset
                    for j in range(len(t_l_imgs[i])): #seq counter
                        
                        t_img = []
                        
                        t_gt = np.zeros([self.seq_length,136])
                        t_gte = np.zeros([self.seq_length,2])
                        
                        t_gt_n = []
                        t_gt_en = []
                        i_t = 0
                        
                        for k in range(len(t_l_imgs[i][j])): #seq size
                            
                            t_img.append(t_l_imgs[i][j][k])
                            t_gt[i_t] = t_l_gt[i][j][k]
                            t_gte[i_t] = t_l_gtE[i][j][k]
                            
                            t_gt_n.append(t_list_gt_names[i][j][k])
                            t_gt_en.append(t_list_gtE_names[i][j][k])
                            
                            i_t+=1
                            
                        self.l_imgs.append(t_img)
                        self.l_gt[indexer] = t_gt
                        self.l_gtE[indexer] = t_gte
                        
                        self.list_gt_names.append(t_gt_n)
                        self.list_gtE_names.append(t_gt_en)
                        
                        indexer+=1
                        
        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_VA = []; l_ldmrk = []; l_nc = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_l = [self.l_gt[index].copy()];label_n =[self.list_gt_names[index]] 
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_l = self.l_gt[index].copy();label_n =self.list_gt_names[index]
        
        for x,labelE,label,ln in zip(x_l,labelE_l,label_l,label_n) : 
            #print(x,labelE,label,ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            if self.onlyFace :    
                #crop the face region
                #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
                if self.ldmrkNumber > 49 : 
                    t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label.copy(),div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
                else : 
                    t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  utils.get_enlarged_bb(the_kp = label.copy(),
                                                           div_x = 3,div_y = 3,images = cv2.imread(x), n_points = 49)#,displacementxy = random.uniform(-.5,.5))
                
                area = (x1,y1, x2,y2)
                tImage =  tImage.crop(area)
                
                label[:self.ldmrkNumber] -= x_min
                label[self.ldmrkNumber:] -= y_min
                
                tImage = tImage.resize((self.imageWidth,self.imageHeight))
                
                label[:self.ldmrkNumber] *= truediv(self.imageWidth,(x2 - x1))
                label[self.ldmrkNumber:] *= truediv(self.imageHeight,(y2 - y1))
                
                
                #now aliging 
                if self.align : 
                    tImageT = utils.PILtoOpenCV(tImage.copy())
                    if self.swap : 
                        ptsSource = torch.tensor([
                            [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                            ])
                        ptsSn = [
                            [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                            ]
                    else : 
                        ptsSource = torch.tensor([
                            [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                            ])
                        ptsSn =[
                            [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                            ]
                        
                        ptsSnFull = np.column_stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:]))
                        ptsSnFull = np.asarray(ptsSnFull,np.float32)
                        
                    ptsSource = ptsSource.numpy()
                    ptsSource = np.asarray(ptsSource,np.float32)
                    
                    if self.useNudget : 
                        trans = nudged.estimate(ptsSn,self.ptsTn)
                        M = np.asarray(trans.get_matrix())[:2,:]
                        #print("Nudged : ",mN,trans.get_scale(),trans.get_rotation())
                    else :
                        #M = cv2.getAffineTransform(ptsSource,self.ptsDst)
                        #_,_,aff  = self.procrustes(ptsSource,self.ptsDst)
                         
                        #print(ptsSource.shape,'-', self.ptsDst.shape)
                        #print(ptsSnFull.shape,'-', self.ptsTnFull.shape)
                        
                        _,_,aff  = self.procrustes(self.ptsTnFull,ptsSnFull)
                        M = aff[:2,:]
                        
                    dst = cv2.warpAffine(tImageT,M,(self.imageWidth,self.imageHeight))
                    
                    #print(np.asarray(ptsSn).shape, np.asarray(self.ptsTn).shape,M.shape)
                    
                    
                    M_full = np.append(M,[[0,0,1]],axis = 0)
                    l_full = np.stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:],np.ones(self.ldmrkNumber)))
                    
                    ldmark = np.matmul(M_full, l_full)
                    
                    if False : 
                        print(ldmark)
                        for i in range(self.ldmrkNumber) :
                            cv2.circle(dst,(int(scale(ldmark[0,i])),int(scale(ldmark[1,i]))),2,(0,255,0) )
                        
                        cv2.imshow('test align',dst)
                        cv2.waitKey(0)
                    
                    label = np.concatenate((ldmark[0],ldmark[1]))
                    tImage = utils.OpenCVtoPIL(dst)
                    
            newChannel = None
            
            if self.wHeatmap : 
                theMiddleName = 'img'
                filePath = x.split(os.sep)
                ifolder = filePath.index(theMiddleName)
                
                print(ifolder)
                image_name = filePath[-1]
                
                annot_name_H = os.path.splitext(image_name)[0]+'.npy'
                
                sDirName = filePath[:ifolder]
                dHeatmaps = '/'.join(sDirName)+'/heatmaps'
                
                finalTargetH = dHeatmaps+'/'+annot_name_H
                print(finalTargetH)
                
                if isfile(finalTargetH) and False: 
                    newChannel  = np.load(finalTargetH)
                    newChannel = Image.fromarray(newChannel)
                else : 
                    checkDirMake(dHeatmaps)
                    
                    tImageTemp = cv2.cvtColor(np.array(tImage),cv2.COLOR_RGB2BGR)
                    #tImageTemp = cv2.imread(x)#tImage.copy()
                    
                    print(len(label),label)
                    
                    b_channel,g_channel,r_channel = tImageTemp[:,:,0],tImageTemp[:,:,1],tImageTemp[:,:,2]
                    newChannel = b_channel.copy(); newChannel[:] = 0
                    
                    t0,t1,t2,t3 = utils.get_bb(label[0:self.ldmrkNumber], label[self.ldmrkNumber:],length=self.ldmrkNumber)
                    
                    l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.05)
                    height, width,_ = tImageTemp.shape
                    
                    wx = t2-t0
                    wy = t3-t1
                
                    scaler = 255/np.max(rv)
                    
                    for iter in range(self.ldmrkNumber) :
                        ix,iy = int(label[iter]),int(label[iter+self.ldmrkNumber])
                        
                        #Now drawing given the center
                        for iter2 in range(len(l_cd)) : 
                            value = int(rv[iter2]*scaler)
                            if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                                newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                    
                    '''tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
                    cv2.imshow("combined",tImage2)
                    cv2.waitKey(0)'''
                    
                    np.save(finalTargetH,newChannel)
                    newChannel = Image.fromarray(newChannel)
            
            if self.augment : 
                sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    flip = RandomHorizontalFlip_WL(1,self.ldmrkNumber)
                    tImage,label,newChannel = flip(tImage,label,newChannel,self.ldmrkNumber)
                elif sel == 2 and not self.align : 
                    rot = RandomRotation_WL(45)
                    tImage,label,newChannel = rot(tImage,label,newChannel,self.ldmrkNumber)
                elif sel == 3 : 
                    occ = Occlusion_WL(1)
                    tImage,label,newChannel = occ(tImage,label,newChannel)
                    
                #random crop
                if not self.align : 
                    rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage,label,newChannel= rc(tImage,label,newChannel)
                
                #additional blurring
                if (np.random.randint(1,3)%2==0) and True : 
                    sel_n = np.random.randint(1,6)
                    #sel_n = 4
                    rc = GeneralNoise_WL(1)
                    tImage,label= rc(tImage,label,sel_n,np.random.randint(0,3))
            
            if self.returnM : 
                if self.swap : 
                    ptsSource = torch.tensor([
                        [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                        ])
                    ptsSn = [
                        [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                        ]
                else : 
                    ptsSource = torch.tensor([
                        [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                        ])
                    ptsSn =[
                        [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                        ]
                    
                    ptsSnFull = np.column_stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:]))
                    ptsSnFull = np.asarray(ptsSnFull,np.float32)
                    
                ptsSource = ptsSource.numpy()
                ptsSource = np.asarray(ptsSource,np.float32)
                
                if self.useNudget : 
                    trans = nudged.estimate(ptsSn,self.ptsTn)
                    M = np.asarray(trans.get_matrix())[:2,:]
                else :
                    #M = cv2.getAffineTransform(ptsSource,self.ptsDst)
                    _,_,aff  = self.procrustes(self.ptsTnFull,ptsSnFull)
                    M = aff[:2,:]    
                
                if False :
                    tImageT = utils.PILtoOpenCV(tImage.copy())
                    dst = cv2.warpAffine(tImageT,M,(self.imageWidth,self.imageHeight))
                    
                    print(np.asarray(ptsSn).shape, np.asarray(self.ptsTn).shape,M.shape)
                    
                    M_full = np.append(M,[[0,0,1]],axis = 0)
                    l_full = np.stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:],np.ones(self.ldmrkNumber)))
                    
                    ldmark = np.matmul(M_full, l_full)
                    print(ldmark)
                    for i in range(self.ldmrkNumber) :
                        cv2.circle(dst,(int(scale(ldmark[0,i])),int(scale(ldmark[1,i]))),2,(0,0,255) )
                    
                    cv2.imshow('test recovered',dst)
                    cv2.waitKey(0)
                    
                    
                
                Minter = self.param2theta(np.append(M,[[0,0,1]],axis = 0), self.imageWidth,self.imageHeight)
                Mt = torch.from_numpy(Minter).float()
            else : 
                Mt = torch.zeros(1)
            
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
            else : 
                tImage = self.transform(tImage)
            
            
            
            
            if not self.wHeatmap : 
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            else : 
                newChannel = transforms.Resize(224)(newChannel)
                newChannel = transforms.ToTensor()(newChannel)
                newChannel = newChannel.sub(125)
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label)); l_nc.append(newChannel)
                #return tImage,torch.FloatTensor(labelE),torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
                
                
                
        if not self.isVideo : 
            if self.wHeatmap : 
                return l_imgs[0], l_VA[0], l_ldmrk[0], l_nc[0], Mt
            else : 
                return l_imgs[0], l_VA[0], l_ldmrk[0], Mt
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            
            if self.wHeatmap :
                #lnc = torch.Tensor(len(l_nc),1,self.imageHeight,self.imageWidth)
                #torch.cat(l_nc, out=lnc)
                lnc = torch.stack(l_nc)
                return lImgs, lVA, lLD, lnc, Mt
            else : 
                return lImgs, lVA, lLD, Mt
                
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    
    def param2theta(self,param, w, h):
        param = np.linalg.inv(param)
        theta = np.zeros([2,3])
        theta[0,0] = param[0,0]
        theta[0,1] = param[0,1]*h/w
        theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
        theta[1,0] = param[1,0]*w/h
        theta[1,1] = param[1,1]
        theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
        return theta
    
    def procrustes(self, X, Y, scaling=True, reflection='best'):
        
        n,m = X.shape
        ny,my = Y.shape
    
        muX = X.mean(0)
        muY = Y.mean(0)
        
        X0 = X - muX
        Y0 = Y - muY
        
        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()
        
        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)
        
        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY
        
        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
        
        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)
        
        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0
            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)
        
        traceTA = s.sum()
        
        if scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY
            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2
            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX
        
        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX
        
        # transformation matrix
        if my < m:
            T = T[:my,:]
        
        c = muX - b*np.dot(muY, T)
        
        #transformation values 
        #tform = {'rotation':T, 'scale':b, 'translation':c}
        tform = np.append(b*T,[c],axis = 0).T
        tform = np.append(tform,[[0,0,1]],axis = 0)
        
        return d, Z, tform
    
    def __len__(self):
        return len(self.l_imgs)













class SEWAFEWReducedLatent(data.Dataset): #return affect on Valence[0], Arousal[1] order
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AFEW"],dir_gt = None, image_size =224, step = 1,split = False, 
                 nSplit = 5, listSplit = [0,1,2,3,4],isVideo = False, seqLength = None, dbType = 0, 
                 returnQuadrant = False, returnWeight = False,useAll = False,
                 splitNumber = None,returnVAQ=False,returnFName = False,isSemaine=False):#dbtype 0 is AFEW, 1 is SEWA
        
        self.dbType = dbType
        
        self.isSemaine = isSemaine
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        self.returnNoisy = False
        self.returnVAQ = returnVAQ
        
        self.returnFName = returnFName
        
        self.curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        if dbType == 0 : 
            featName = "FT-AF0-0-16-16-Den"
        else : 
            featName = "FT-SW0-0-16-16-Den"
            
        if self.isSemaine : 
            featName = "FT-SEM0-0-16-16-Den"
        
        if useAll :
            featName+="-UA" 
            
        
        featName+="-z"
        
        self.returnWeight = returnWeight
        
        if self.returnWeight :
            name = 'VA-Train-'+str(listSplit[0])+'.npy' 
            if self.dbType == 1 : 
                name='S-'+name
                
            if isSemaine : 
                name = 'SE-VA-Train-'+str(listSplit[0])+'.npy' 
                
            weight = np.load(rootDir+"/DST-SE-AF/"+name).astype('float')+1
            sum = weight.sum(0)
            
            weight = (weight/sum)
            #print('1',weight)
            
            weight = 1/weight
            #print('2',weight)
            
            sum = weight.sum(0)
            weight = weight/sum
            #print('3',weight)
            
            self.weight =  weight
        
        self.returnQ = returnQuadrant
        
            
        list_gt = []
        list_labels_tE = []
        
        counter_image = 0
        
        annotE_name = 'annot2'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        list_missing = []
        
        
        for data in data_list : 
            print(("Opening "+data))
            
            fullDir = self.curDir +data+"/"
            listFolder = os.listdir(fullDir)
            listFolder.sort()
            
            for tempx in range(len(listFolder)):
                f = listFolder[tempx]
                fullPath = os.path.join(fullDir,f)
                #print('opening fullpath',fullPath)
                if os.path.isdir(fullPath): # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    #c_image,c_ldmark = 0,0
                    
                    if self.dbType == 1 : #we directly get the VA file in case of sewa  
                        #first get the valence 
                        valFile = fullPath+"/valence/"+f+"_Valence_A_Aligned.csv"
                        aroFile = fullPath+"/arousal/"+f+"_Arousal_A_Aligned.csv"
                        
                        list_labels_tE.append([valFile,aroFile])
                        #print(valFile,aroFile)
                    
                    #print('fp ',fullPath)
                    for sub_f in file_walker.walk(fullPath):
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            #print(sub_f.name)
                            if(sub_f.name == featName): #Else it is the image
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                    
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
                                c_ldmrk = len(list_dta)
        
                            elif (sub_f.name == annotE_name) :
                                if self.dbType == 0 : 
                                    #If that's annot, add to labels_t
                                    for sub_sub_f in sub_f.walk(): #this is the data
                                        if(".npy" not in sub_sub_f.full_path):
                                            list_dta.append(sub_sub_f.full_path)
                                    list_labels_tE.append(sorted(list_dta))
                                    
                                    
        self.length = counter_image
        print("Now opening keylabels")
         
        list_labelsEN = []
        list_labelsE = []
        
        for ix in range(len(list_labels_tE)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            
            lbl_2 = [] #Per folder
            lbl_n2 = [] #Per folder
            
            if self.dbType == 1 : #sewa  
                #print(list_labels_t[ix][0])
                valFile = np.asarray(readCSV(list_labels_tE[ix][0]))
                aroFile = np.asarray(readCSV(list_labels_tE[ix][1]))
                
                lbl_n2.append(list_labels_tE[ix][0])
                lbl_2 = np.column_stack((valFile,aroFile))
            else : 
                for jx in range(len (list_labels_tE[ix])): #lbl_sub in lbl :
                    
                    #print(os.path.basename(list_gt[ix][jx]))
                    #print(os.path.basename(list_labels_t[ix][jx]))
                    #print(os.path.basename(list_labels_tE[ix][jx]))
                    
                    if self.dbType == 0 : 
                        lbl_subE = list_labels_tE[ix][jx]
                        if ('aro' in lbl_subE) : 
                            x = []
                            #print(lbl_sub)
                            with open(lbl_subE) as file:
                                data2 = [re.split(r'\t+',l.strip()) for l in file]
                            for i in range(len(data2)) :
                                temp = [ float(j) for j in data2[i][0].split()]
                                temp.reverse() #to give the valence first. then arousal
                                x.append(temp)
                            
                            lbl_2.append(np.array(x).flatten('F'))
                            lbl_n2.append(lbl_subE)
                
            
            list_labelsEN.append(lbl_n2)
            list_labelsE.append(lbl_2)
        
            
        t_l_imgs = []
        t_l_gtE = []
        
        t_list_gtE_names = []
        
        #print(list_labelsEN)
        print(len(list_labelsE))
        print(len(list_labelsE[0]))
        print(len(list_labelsE[0][0]))
        
        print(list_labelsE[0][0])
        
        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gtE_names = []
                indexer = 0
                
                list_ground_truthE = np.zeros([len(list_gt[i]),2])
                
                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])
                    #print(list_labelsEN)
                    if self.dbType == 0 : 
                        list_gtE_names.append(list_labelsEN[i][j])
                    else : 
                        list_gtE_names.append(list_labelsEN[i][0])
                    #print(list_labelsEN[i])
                    
                    '''if len(list_labels[i][j] < 1): 
                        print(list_labels[i][j])'''
                    #print(len(list_labels[i][j]))
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_imgs.append(list_images)
                t_l_gtE.append(list_ground_truthE)
                t_list_gtE_names.append(list_gtE_names)
        else : 
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;
                    list_gtE_names = []
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize   
                        
                        
                        
                        temp = []
                        tmpn2 = []
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z]) 
                            #print(list_labelsE[i][z])
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                            
                            
                            if self.dbType == 0 : 
                                #list_gtE_names.append(list_labelsEN[i][j])
                                tmpn2.append(list_labelsEN[i][j])
                            else : 
                                #list_gtE_names.append(list_labelsEN[i][0])
                                tmpn2.append(list_labelsEN[i][0])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truthE[indexer] = temp3
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gtE_names.append(list_gtE_names)
                
        self.l_imgs = []
        self.l_gtE = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if split :
            indexer = 0 
            
            self.l_gtE = []
                
            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                
                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): 
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])
                            
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1
                            
                else : 
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter
                        
                            t_img = []
                            t_gtE = []
                            t_gt_EN = []
                            tmp = 0
                            
                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1
                                
                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gtE.append(t_gtE)
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1
                    
                print(len(self.l_imgs))

        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_ldmrk = []; l_VA = []; l_nc = []; l_qdrnt = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        
        if self.returnFName : 
            l_fname = []
            
        if self.returnNoisy : 
            l_nimgs = []
        
        if self.returnWeight : 
            l_weights = []
        
        if self.returnVAQ : 
            l_vaq = []
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_n =[self.list_gtE_names[index]]
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_n =self.list_gtE_names[index]
        
        
        #print('label n ',label_n)
        for x,labelE,ln in zip(x_l,labelE_l,label_n) : 
            tmp = np.load(x);#tImage = np.load(x) #Image.open(x).convert("RGB")
            
            if self.returnFName : 
                l_fname.append(x)
            
            reduce = True  
            if reduce : 
                tImage = tmp['z'][:64]
            else : 
                tImage=tmp['z']
            
            if self.returnVAQ: 
                vaq = torch.from_numpy(tmp['vaq'])
                l_vaq.append(vaq)
            
            #tImage = np.load(x)
            nImage = tImage.copy()
                    
            label = torch.zeros(1)
            Mt = torch.zeros(1)
            
            
            tImage = torch.from_numpy(tImage)
            if self.returnNoisy : 
                nImage = torch.from_numpy(nImage)
            
            
            #print('shap e: ', tImage.shape)
            
            
            l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            if self.returnNoisy : 
                l_nimgs.append(nImage)
            
            
            
            if self.returnQ : 
                if self.dbType == 1 :
                    min = 0; max = 1;
                elif self.isSemaine == 1:
                    min = -1; max = 1;
                else : 
                    min = -10; max = 10;
                
                l_qdrnt.append(toQuadrant(labelE, min, max, toOneHot=False))
                
            if self.returnWeight :
                v = labelE[0] 
                a = labelE[0]
                
                if self.dbType == 1 :#sewa 
                    v = v*10+1
                    a = a*10+1
                elif self.isSemaine == 1 : 
                    v = v*10+10
                    a = a*10+10
                else :
                    v = v+10
                    a = a+10
                
                v,a = int(v),int(a)
                l_weights.append([self.weight[v,0],self.weight[a,1]])
                
            l_nc.append(ln)
            
        
        #print('lnc : ',l_nc)
        if not self.isVideo : 
            if self.returnQ : 
                if self.returnNoisy :
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0],l_qdrnt[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0],l_qdrnt[0]]
            else :
                if self.returnNoisy : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0]]
                    
            if self.returnWeight :
                res.append(torch.tensor(l_weights[0]))
                
                
            if self.returnVAQ : 
                res.append(torch.tensor(l_vaq[0]))
                #res.append(l_vaq)
                
                
            if self.returnFName: 
                res.append(l_fname[0])
                
                
            return res 
        else : 
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(l_VA)
            l_qdrnt = torch.tensor((l_qdrnt))
            
            if self.returnQ : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt]
            else : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc]
                    
            if self.returnWeight : 
                l_weights = torch.tensor(l_weights)
                res.append(l_weights)
                
            if self.returnVAQ : 
                l_vaq = torch.tensor(l_vaq)
                res.append(l_vaq)
                
            if self.returnFName: 
                res.append(l_fname)
                
            return res 
    
    def __len__(self):
        return len(self.l_imgs)
    

class SEWAFEWReduced(data.Dataset): #return affect on Valence[0], Arousal[1] order
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AFEW"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,split = False, 
                 nSplit = 5, listSplit = [0,1,2,3,4],isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = False, isSemaine = False):#dbtype 0 is AFEW, 1 is SEWA
        
        self.dbType = dbType
        
        self.isSemaine = isSemaine
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
        self.returnNoisy = returnNoisy
        self.returnWeight = returnWeight
        
        if self.returnWeight :
            name = 'VA-Train-'+str(listSplit[0])+'.npy' 
            if self.dbType == 1 : 
                name='S-'+name
            if isSemaine : 
                name = 'SE-VA-Train-'+str(listSplit[0])+'.npy' 
            
            print('weight',name)
            weight = np.load(rootDir+"/DST-SE-AF/"+name).astype('float')+1
            sum = weight.sum(0)
            
            weight = (weight/sum)
            #print('1',weight)
            
            weight = 1/weight
            #print('2',weight)
            
            sum = weight.sum(0)
            weight = weight/sum
            #print('3',weight)
            "just tesing for the latencyh if its possible. "
            self.weight =  weight
        
        self.returnQ = returnQuadrant
        
        if self.augment : 
            self.flip = RandomHorizontalFlip(1)
            self.rot = RandomRotation(45)
            self.occ = Occlusion(1)
            self.rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
            
        if self.returnNoisy : 
            self.gn = GeneralNoise(1)
            self.occ = Occlusion(1)
            
        list_gt = []
        list_labels_tE = []
        
        counter_image = 0
        
        annotE_name = 'annot2'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        list_missing = []
        
        
        for data in data_list : 
            print(("Opening "+data))
            
            fullDir = self.curDir +data+"/"
            listFolder = os.listdir(fullDir)
            listFolder.sort()
            
            for tempx in range(len(listFolder)):
                f = listFolder[tempx]
                fullPath = os.path.join(fullDir,f)
                #print('opening fullpath',fullPath)
                if os.path.isdir(fullPath): # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    #c_image,c_ldmark = 0,0
                    
                    if self.dbType == 1 : #we directly get the VA file in case of sewa  
                        #first get the valence 
                        valFile = fullPath+"/valence/"+f+"_Valence_A_Aligned.csv"
                        aroFile = fullPath+"/arousal/"+f+"_Arousal_A_Aligned.csv"
                        
                        list_labels_tE.append([valFile,aroFile])
                        #print(valFile,aroFile)
                        
                    for sub_f in file_walker.walk(fullPath):
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            #print(sub_f.name)
                            if(sub_f.name == 'img-128'): #Else it is the image
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
                                c_ldmrk = len(list_dta)
        
                            elif (sub_f.name == annotE_name) :
                                if self.dbType == 0 : 
                                    #If that's annot, add to labels_t
                                    for sub_sub_f in sub_f.walk(): #this is the data
                                        if(".npy" not in sub_sub_f.full_path):
                                            list_dta.append(sub_sub_f.full_path)
                                    list_labels_tE.append(sorted(list_dta))
                                    
                                    
        self.length = counter_image
        print("Now opening keylabels")
         
        list_labelsEN = []
        list_labelsE = []
        
        for ix in range(len(list_labels_tE)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            
            lbl_2 = [] #Per folder
            lbl_n2 = [] #Per folder
            
            if self.dbType == 1 : #sewa  
                #print(list_labels_t[ix][0])
                valFile = np.asarray(readCSV(list_labels_tE[ix][0]))
                aroFile = np.asarray(readCSV(list_labels_tE[ix][1]))
                
                #lbl_n2.append(list_labels_tE[ix][0])
                for it in range(1,len(valFile)+1): 
                    dir,_ = os.path.split(list_labels_tE[ix][0])
                    newName = str(it).zfill(6)+'.tmp'
                    
                    lbl_n2.append(os.path.join(dir,newName))
                
                lbl_2 = np.column_stack((valFile,aroFile))
            else : 
                for jx in range(len (list_labels_tE[ix])): #lbl_sub in lbl :
                    
                    #print(os.path.basename(list_gt[ix][jx]))
                    #print(os.path.basename(list_labels_t[ix][jx]))
                    #print(os.path.basename(list_labels_tE[ix][jx]))
                    
                    if self.dbType == 0 : 
                        lbl_subE = list_labels_tE[ix][jx]
                        if ('aro' in lbl_subE) : 
                            x = []
                            #print(lbl_sub)
                            with open(lbl_subE) as file:
                                data2 = [re.split(r'\t+',l.strip()) for l in file]
                            for i in range(len(data2)) :
                                temp = [ float(j) for j in data2[i][0].split()]
                                temp.reverse() #to give the valence first. then arousal
                                x.append(temp)
                            
                            lbl_2.append(np.array(x).flatten('F'))
                            lbl_n2.append(lbl_subE)
                
            
            list_labelsEN.append(lbl_n2)
            list_labelsE.append(lbl_2)
            
        t_l_imgs = []
        t_l_gtE = []
        
        t_list_gtE_names = []
        
        #print(list_labelsEN)
        
        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gtE_names = []
                indexer = 0
                
                list_ground_truthE = np.zeros([len(list_gt[i]),2])
                
                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])
                    #print(list_labelsEN)
                    if self.dbType == 0 : 
                        list_gtE_names.append(list_labelsEN[i][j])
                    else : 
                        #list_gtE_names.append(list_labelsEN[i][0])
                        list_gtE_names.append(list_labelsEN[i][j])
                    #print(list_labelsEN[i])
                    
                    '''if len(list_labels[i][j] < 1): 
                        print(list_labels[i][j])'''
                    #print(len(list_labels[i][j]))
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_imgs.append(list_images)
                t_l_gtE.append(list_ground_truthE)
                t_list_gtE_names.append(list_gtE_names)
        else : 
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;
                    list_gtE_names = []
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize   
                        
                        
                        
                        temp = []
                        tmpn2 = []
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z]) 
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                            
                            
                            if self.dbType == 0 : 
                                #list_gtE_names.append(list_labelsEN[i][j])
                                tmpn2.append(list_labelsEN[i][j])
                            else : 
                                #list_gtE_names.append(list_labelsEN[i][0])
                                tmpn2.append(list_labelsEN[i][0])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truthE[indexer] = temp3
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gtE_names.append(list_gtE_names)
                
        
        self.l_imgs = []
        self.l_gtE = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if split :
            
            indexer = 0 
            
            self.l_gtE = []
            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                
                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): 
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])
                            
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1
                            
                else : 
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter
                        
                            t_img = []
                            t_gtE = []
                            t_gt_EN = []
                            tmp = 0
                            
                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1
                                
                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gtE.append(t_gtE)
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1
                    
                print(len(self.l_imgs))
                    
            self.l_gtE = np.asarray(self.l_gtE)
        else :
            if not self.isVideo :
                self.l_gtE = np.zeros([counter_image,2])
                indexer = 0
                
                
                for i in range(len(t_l_imgs)): 
                    for j in range(len(t_l_imgs[i])): 
                        self.l_imgs.append(t_l_imgs[i][j])
                        print(i,j,'-',len(t_l_imgs[i]))
                        self.l_gtE[indexer] = t_l_gtE[i][j]
                        
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1
                    
            else : 
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])
                
                indexer = 0
                
                for i in range(len(t_l_imgs)): #dataset
                    for j in range(len(t_l_imgs[i])): #seq counter
                        
                        t_img = []
                        
                        t_gte = np.zeros([self.seq_length,2])
                        
                        t_gt_n = []
                        t_gt_en = []
                        i_t = 0
                        
                        for k in range(len(t_l_imgs[i][j])): #seq size
                            
                            t_img.append(t_l_imgs[i][j][k])
                            t_gte[i_t] = t_l_gtE[i][j][k]
                            
                            t_gt_en.append(t_list_gtE_names[i][j][k])
                            
                            i_t+=1
                            
                        self.l_imgs.append(t_img)
                        self.l_gtE[indexer] = t_gte
                        self.list_gtE_names.append(t_gt_en)
                        
                        indexer+=1
                        
        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_ldmrk = []; l_VA = []; l_nc = []; l_qdrnt = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        if self.returnNoisy : 
            l_nimgs = []
        
        if self.returnWeight : 
            l_weights = []
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_n =[self.list_gtE_names[index]] 
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_n =self.list_gtE_names[index]
        
        
        #print('label n ',label_n)
        for x,labelE,ln in zip(x_l,labelE_l,label_n) : 
            #print(x,labelE,label,ln)
            #print('label : ',ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            newChannel = None
            
            if self.augment : 
                
                if self.returnNoisy :
                    sel = np.random.randint(0,3) #Skip occlusion as noise
                else : 
                    sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    #flip = RandomHorizontalFlip_WL(1)
                    #tImage,label,newChannel = flip(tImage,label,newChannel)
                    #flip = RandomHorizontalFlip(1)
                    tImage = self.flip(tImage)
                elif sel == 2 : 
                    #rot = RandomRotation_WL(45)
                    #tImage,label,newChannel = rot(tImage,label,newChannel)
                    #rot = RandomRotation(45)
                    tImage = self.rot(tImage)
                elif sel == 3 : 
                    #occ = Occlusion_WL(1)
                    #tImage,label,newChannel = occ(tImage,label,newChannel)
                    #occ = Occlusion(1)
                    tImage = self.occ(tImage)
                    
                #random crop
                if (np.random.randint(1,3)%2==0) : 
                    #rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    #tImage,label,newChannel= rc(tImage,label,newChannel)
                    
                    #rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage= self.rc(tImage)
                
            if self.returnNoisy :
                nImage = tImage.copy()
            
                #additional blurring
                if (np.random.randint(1,3)%2==0): 
                    #sel_n = np.random.randint(1,6)
                    sel_n = np.random.randint(1,7)
                    
                    #sel_n = 4
                    #gn = GeneralNoise_WL(1)
                    #tImage,label= gn(tImage,label,sel_n,np.random.randint(0,3))
                    
                    if sel_n > 5 : 
                        #occ = Occlusion(1)
                        nImage = self.occ(nImage)
                    else :
                        #rc = GeneralNoise(1)
                        #tImage = rc(tImage,sel_n,np.random.randint(0,3))
                        nImage = self.gn(nImage,sel_n,np.random.randint(0,3))
                    
            label = torch.zeros(1)
            Mt = torch.zeros(1)
            
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
                if self.returnNoisy : 
                    nImage = self.transformInternal(nImage)
            else : 
                tImage = self.transform(tImage)
                if self.returnNoisy : 
                    nImage = self.transform(nImage)
            
            
            
            l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            if self.returnNoisy : 
                l_nimgs.append(nImage)
            
            if self.returnQ : 
                if self.dbType == 1 :
                    min = 0; max = 1;
                elif self.isSemaine == 1:
                    min = -1; max = 1;
                else : 
                    min = -10; max = 10;
                
                l_qdrnt.append(toQuadrant(labelE, min, max, toOneHot=False))
                
            if self.returnWeight :
                v = labelE[0] 
                a = labelE[0]
                
                if self.dbType == 1 :#sewa 
                    v = v*10+1
                    a = a*10+1
                elif self.isSemaine == 1 : 
                    v = v*10+10
                    a = a*10+10
                else :
                    v = v+10
                    a = a+10
                
                v,a = int(v),int(a)
                '''print('the v :{} a : {} db : {}'.format(v,a,self.dbType))
                print(self.weight)
                print(self.weight.shape)'''
                l_weights.append([self.weight[v,0],self.weight[a,1]])
                
            l_nc.append(ln)
        
        #print('lnc : ',l_nc)
        if not self.isVideo : 
            if self.returnQ : 
                if self.returnNoisy :
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0],l_qdrnt[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0],l_qdrnt[0]]
            else :
                if self.returnNoisy : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,l_nc[0]]
                    
            if self.returnWeight :
                res.append(torch.tensor(l_weights[0]))
                
            return res 
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(l_VA)
            l_qdrnt = torch.tensor((l_qdrnt))
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            if self.returnQ : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt]
            else : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc]
                    
            if self.returnWeight : 
                l_weights = torch.tensor(l_weights)
                res.append(l_weights)
                
            return res 
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    
    def param2theta(self,param, w, h):
        param = np.linalg.inv(param)
        theta = np.zeros([2,3])
        theta[0,0] = param[0,0]
        theta[0,1] = param[0,1]*h/w
        theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
        theta[1,0] = param[1,0]*w/h
        theta[1,1] = param[1,1]
        theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
        return theta
    
    def procrustes(self, X, Y, scaling=True, reflection='best'):
        
        n,m = X.shape
        ny,my = Y.shape
    
        muX = X.mean(0)
        muY = Y.mean(0)
        
        X0 = X - muX
        Y0 = Y - muY
        
        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()
        
        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)
        
        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY
        
        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
        
        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)
        
        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0
            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)
        
        traceTA = s.sum()
        
        if scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY
            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2
            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX
        
        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX
        
        # transformation matrix
        if my < m:
            T = T[:my,:]
        
        c = muX - b*np.dot(muY, T)
        
        #transformation values 
        #tform = {'rotation':T, 'scale':b, 'translation':c}
        tform = np.append(b*T,[c],axis = 0).T
        tform = np.append(tform,[[0,0,1]],axis = 0)
        
        return d, Z, tform
    
    def __len__(self):
        return len(self.l_imgs)



def convertName(input):
    number = int(re.search(r'\d+', input).group())
    if 'train' in input : 
        return number
    elif 'dev' in input :
        return 10+number
    elif  'test' in input :
        return 20+number

def cropImage():
    
    batch_size = 20
    image_size = 224
    isVideo = False
    doConversion = False
    lndmrkNumber =68
    #lndmarkNumber = 49
    
    isSewa = False
    
    desireS = 224
    smll = desireS!=224
    
    ratio = truediv(desireS,224)
    
    
    if ratio : 
        displaySize = str(128)
    else : 
        displaySize = str(image_size)
    
    err_denoised = curDir+"de-label-"+'semaine'+".txt"
    checkDirMake(os.path.dirname(err_denoised))
    print('file of denoising : ',err_denoised)
    fileOfDen = open(err_denoised,'w')
    fileOfDen.close()
    
    #theDataSet = "AFEW-VA-Small"
    #theDataSet = "AFEW-VA-Fixed"
    #theDataSet = "SEWA-small"
    #theDataSet = "SEWA"
    
    theDataSet = "Sem-Short"
    
    
    oriDir = '/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/'+theDataSet
    #oriDir = '/media/deckyal/INT-2TB/comparisons/'+theDataSet + "/" + str(theNoiseType)+"-"+str(theNoiseParam)+'/'
    
    targetDir = '/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/'+theDataSet+'-ext'
    checkDirMake(targetDir)
 
    data_transform   = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ID = ImageDataset(data_list = [theDataSet],onlyFace=True,transform=data_transform,image_size=image_size
                            ,injectedLink = oriDir,isVideo = isVideo,giveCroppedFace=True,
                            annotName='annot',lndmarkNumber=lndmrkNumber,isSewa = isSewa)
    #annotName = annotOri
    dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = False)
    
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    notNeutral = 0
    list_nn = []
    list_name_nn = []
    
    print('inside',len(dataloader))
    
    
    
    GD = GeneralDAEX(nClass = 3)

    dircl1 = '/home/deckyal/eclipse-workspace/FaceTracking/src/toBeUsedT-5Aug/'+'Mix3-combineAE.pt'
    dircl2 = '/home/deckyal/eclipse-workspace/FaceTracking/src/toBeUsedT-5Aug/'+'Mix3-combineCL.pt'
    outDir = "mix3-"
    model_lg = LogisticRegression(512, 3)
        
    
    netAEC = DAEE()
    netAEC.load_state_dict(torch.load(dircl1))
    netAEC = netAEC.cuda()
    netAEC.eval()
    
    #theDataSetOut = theDataVideo+outDir
    model_lg.load_state_dict(torch.load(dircl2))
    model_lg = model_lg.cuda()
    model_lg.eval()
    
    #print(netAEC.fce.weight)
    print(model_lg.linear2.weight)
    #exit(0)
    isVideo = False
    #exit(0)
    data_transform   = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
    ])

    
    # Plot some training images
    for inside in dataloader : # = next(iter(dataloader))
                
        print(len(inside))
        real_batch,gt,cr,x,gtcr  = inside[0],inside[1],inside[2],inside[3],inside[4]
        if isSewa :
            gtcr2 = inside[5]
        else : 
            gtcr2 = gtcr
        
        print(real_batch.size())
        for imgs,gts,imgcrs,fileName,gtcrs,gts2 in zip(real_batch.cuda(),gt.numpy(),cr.cuda(),x,gtcr.numpy(),gtcr2.numpy()): 
            
            print(fileName)
            
            #first save the original image 
            
            #now getting the name and file path 
            filePath = fileName.split(os.sep)
            annotPath = copy.copy(filePath)
            if isSewa : 
                annotPathSewa = copy.copy(filePath)
                
            filePathCleaned = copy.copy(filePath)
            
            filePath[-2]+='-'+displaySize
            filePathCleaned[-2]+='-'+displaySize+'-C'
            if isSewa : 
                annotPath[-2]='annotOri-'+displaySize
                annotPathSewa[-2]='annot-'+displaySize
            else : 
                annotPath[-2]='annot-'+displaySize
            
            newFilePath = '/'.join(filePath[:-1])
            newAnnotPath = '/'.join(annotPath[:-1])
            if isSewa : 
                newAnnotPathSewa = '/'.join(annotPathSewa[:-1])
            newClFilePath = '/'.join(filePathCleaned[:-1])
            #print(filePath,annotPath)
            print(newFilePath, newAnnotPath)
            #ifolder = filePath.index(theDataVideo)
            
            image_name = filePath[-1]
            annot_name = os.path.splitext(image_name)[0]+'.pts'
            
            '''if isVideo :  
                middle = filePath[ifolder+2:-2]
                print(middle)
                middle = '/'.join(middle)
                
                finalTargetPathI = targetDir+middle+'/img/'
                finalTargetPathA = targetDir+middle+'/annot/'
            else : 
                finalTargetPathI = targetDir+'img/'
                finalTargetPathA = targetDir+'annot/' '''
                
            checkDirMake(newFilePath)
            checkDirMake(newAnnotPath)
            if isSewa : 
                checkDirMake(newAnnotPathSewa)
            checkDirMake(newClFilePath)
            
            finalTargetImage = newFilePath+'/'+image_name
            finalTargetImageCl = newClFilePath+'/'+image_name
            finalTargetAnnot = newAnnotPath+'/'+annot_name
            if isSewa : 
                finalTargetAnnotSewa = newAnnotPathSewa+'/'+annot_name
            
            
            theOri = unorm(imgcrs.detach().cpu()).numpy()*255
            theOri = cv2.cvtColor(theOri.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
            
            if smll :
                theOri = cv2.resize(theOri,(128,128))
            cv2.imwrite(finalTargetImage,theOri)
            
            if smll : 
                gtcrs[:lndmrkNumber] *= ratio
                gtcrs[lndmrkNumber:] *= ratio
                
                if isSewa : 
                    gts2[:68] *= ratio
                    gts2[68:] *= ratio
            
            write_kp_file(finalTargetAnnot,gtcrs,length = lndmrkNumber)
            
            if isSewa : 
                write_kp_file(finalTargetAnnotSewa,gts2,length = 68)
            
            
            #print(gtcrs)
            
            #Now see the result back
            
            r_image = cv2.imread(finalTargetImage)
            
            print(finalTargetAnnot)
            predicted = utils.read_kp_file(finalTargetAnnot, True)
            for z22 in range(lndmrkNumber) :
                #print(z22)
                cv2.circle(r_image,(int(predicted[z22]),int(predicted[z22+lndmrkNumber])),2,(0,255,0))
                
            if isSewa:
                predicted2 = utils.read_kp_file(finalTargetAnnotSewa, True)
                for z22 in range(68) :
                    cv2.circle(r_image,(int(predicted2[z22]),int(predicted2[z22+68])),2,(255,255,255))
                    
            cv2.imshow('test',r_image)
            cv2.waitKey(1)
            
            #exit(0)
            
            #second get the cleaned one 
            
            #if cl_type == 1 : 
            recon_batch,xe = netAEC(imgs.unsqueeze(0))
            #else :  
            #    xe = netAEC(imgs.unsqueeze(0))
                
            labels = model_lg(xe)
            x, y = torch.max(labels, 1)
            
            ll = y.cpu()[0]
            
            print('res',ll)
            
            #res = GD.forward(imgs.unsqueeze(0), y[0])[0].detach().cpu()
            res = GD.forward(imgcrs.unsqueeze(0), y[0])[0].detach().cpu()
            
            theRest = unorm(res).numpy()*255
            print(theRest.shape)
            theRest = cv2.cvtColor(theRest.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
            
            
            if smll : 
                theRest = cv2.resize(theRest,(128,128))
            
            
            theOri = unorm(imgs.detach().cpu()).numpy()*255
            print(theOri.shape)
            theOri = cv2.cvtColor(theOri.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
            
            cv2.imshow('theori',theRest)
            cv2.waitKey(1)
            
            
            cv2.imwrite(finalTargetImageCl,theRest)
            #third save the cleaned one
            
            #exit(0)
            
            '''
            #print(theRest.shape)
            
            theImage = theRest
            
            #now getting the name and file path 
            filePath = fileName.split(os.sep)
            ifolder = filePath.index(theDataVideo)
            
            image_name = filePath[-1]
            annot_name = os.path.splitext(image_name)[0]+'.pts'
            
            if isVideo :  
                middle = filePath[ifolder+2:-2]
                print(middle)
                middle = '/'.join(middle)
                
                finalTargetPathI = targetDir+middle+'/img/'
                finalTargetPathA = targetDir+middle+'/annot/'
            else : 
                finalTargetPathI = targetDir+'img/'
                finalTargetPathA = targetDir+'annot/'
                
            checkDirMake(finalTargetPathI)
            checkDirMake(finalTargetPathA)
            
            finalTargetImage = finalTargetPathI+image_name
            finalTargetAnnot = finalTargetPathA+annot_name
            
            print(finalTargetImage,finalTargetAnnot)'''
            
            
            if ll != 0 or True: 
                if ll != 0:
                    notNeutral+=1
                    list_nn.append(ll)
                    list_name_nn.append(finalTargetImage)
            
            
                fileOfDen = open(err_denoised,'a')
                fileOfDen.write(str(int(ll))+','+finalTargetImage+"\n")
                fileOfDen.close()
                
                print('status : ',ll)
                '''
                cv2.imshow('ori',theOri)
                cv2.waitKey(0)
                cv2.imshow('after',theRest)
                cv2.waitKey(0)'''
                
            print(y,labels)
            
    
    print("not neutral count : ",notNeutral)            



def getDistributionAC():
        
    import matplotlib.pyplot as plt 
        
    targetDir = '/home/deckyal/eclipse-workspace/FaceTracking/FaceTracking-NR/StarGAN_Collections/stargan-master/distribution/'
        
    
    tname = "AC"
    
            
    image_size = 112
    batch_size = 20000
    
    transform = transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    if False :
         
        a = np.array(range(20))
        v = np.array(range(20))
        
        tx = np.array(range(20))
        
        for i in range(5) : 
            
            z = np.load(targetDir+'VA-Train-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv 
            
            z = np.load(targetDir+'VA-Test-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv
            
            
        
        fig = plt.figure()
    
        ax = plt.subplot(2, 2, 1)
        ax.bar(tx,a)
        ax.set_title('a')
        
        ax = plt.subplot(2, 2, 2)
        ax.bar(tx,v)
        ax.set_title('v')
        
        plt.show()
        
        
        
        #print(a)
        #print(v)
        exit(0)
    
    
    
    ID = AFFChallenge(data_list = ["AffectChallenge"],mode = 'Train',onlyFace = True, image_size =112, 
                 transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = False)
    
    VD = AFFChallenge(data_list = ["AffectChallenge"],mode = 'Val',onlyFace = True, image_size =112, 
                 transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = False)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    
    
    dataloaderTrn = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = False)
    dataloaderVal = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)

    
    listV = np.array(range(0,21))
    listA = np.array(range(0,21))
    
    listVx = np.array(range(0,21))
    listAx = np.array(range(0,21))
    
    listVt = np.array(range(0,21))
    listAt = np.array(range(0,21))
        
    #for real_batch,vas,gt,M,_ in (dataloaderTrn) :
    x = 0 
    for lImgs,vas,gt,M,ex in (dataloaderTrn) :
        
        for va in vas : 
            print(x,len(dataloaderTrn)*batch_size)
            #print(ex,gt,M)
            #print(va,vas)
            lva = (va.cpu().numpy()) * 10+10
            name = 'AC-Train'
            print(va)
            print(lva)
            listV[int(round(lva[0]))]+=1
            listA[int(round(lva[1]))]+=1
        
            x+=1
    
    x = 0
    
    print(listV,listA)
    np.save(targetDir+name+'.npy',np.column_stack((listV,listA)))
    
    for real_batch,vas,gt,M,ex in (dataloaderVal) :
        for va in vas : 
            print(x,len(dataloaderVal)*batch_size) 
            lva = (va.cpu().numpy()) * 10+10
            name = 'AC-Test-'
            listVt[int(round(lva[0]))]+=1
            listAt[int(round(lva[1]))]+=1
            x+=1
        
    print(listVt,listAt)
    np.save(targetDir+name+'.npy',np.column_stack((listVt,listAt)))
    
    
    '''fig, ax = plt.subplots(nrows=1, ncols=2)
    
    for row in ax:
        for col in row:
            col.plot(x, y)'''
    
    fig = plt.figure()

    ax = plt.subplot(2, 2, 1)
    ax.bar(listVx,listV)
    ax.set_title('v train')
    
    ax = plt.subplot(2, 2, 2)
    ax.bar(listAx,listA)
    ax.set_title('A train')
    
    ax = plt.subplot(2, 2, 3)
    ax.bar(listVx,listVt)
    ax.set_title('v test')
    
    ax = plt.subplot(2, 2, 4)
    ax.bar(listAx,listAt)
    ax.set_title('A test')
    
    #plt.show()
    plt.savefig(tname+".png")
        
    exit(0)
    
    

def getDistribution():
        
    import matplotlib.pyplot as plt 
        
    targetDir = '/home/deckyal/eclipse-workspace/FaceTracking/FaceTracking-NR/StarGAN_Collections/stargan-master/distribution/'
        
    
    isAFEW = True
    isSemaine = True
    
    name = "AFEW"
    if not isAFEW : 
        name = "SEWA"
    
            
    image_size = 224
    batch_size = 1000#5000
    
    transform = transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    
    
    if False : 
        a = np.array(range(20))
        v = np.array(range(20))
        
        tx = np.array(range(20))
        
        for i in range(5) : 
            
            z = np.load(targetDir+'VA-Train-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv 
            
            z = np.load(targetDir+'VA-Test-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv
            
            
        
        fig = plt.figure()
    
        ax = plt.subplot(2, 2, 1)
        ax.bar(tx,a)
        ax.set_title('a')
        
        ax = plt.subplot(2, 2, 2)
        ax.bar(tx,v)
        ax.set_title('v')
        
        plt.show()
        
        
        
        #print(a)
        #print(v)
        exit(0)
    
    
    for split in range(5) : 
        minA,minV = 9999,9999
        maxA,maxV = -9999,-9999
        
        #split = 1
        multi_gpu = False
        testSplit = split
        print("Test split " , testSplit)
        nSplit = 5
        listSplit = []
        for i in range(nSplit):
            if i!=testSplit : 
                listSplit.append(i)
        print(listSplit)
        
        #sem short
        #sem small
        
        if not isAFEW : 
        
            ID = SEWAFEWReduced(data_list = ["SEWA-small"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=listSplit,
                      isVideo=False, seqLength = 6,dbType = 1)
            
            VD = SEWAFEWReduced(data_list = ["SEWA-small"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[testSplit],
                      isVideo=False, seqLength = 6,dbType = 1)
        else : 
        
            
            ''' ID = SEWAFEWReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=listSplit,
                      isVideo=False, seqLength = 6,dbType = 0)
            
            VD = SEWAFEWReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[testSplit],
                      isVideo=False, seqLength = 6,dbType = 0)
            '''
            if isSemaine : 
                ID = SEWAFEWReduced(data_list = ["Sem-Short"], dir_gt = None, onlyFace = True, image_size = image_size, 
                          transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=listSplit,
                          isVideo=False, seqLength = 6,dbType = 0)
                
                VD = SEWAFEWReduced(data_list = ["Sem-Short"], dir_gt = None, onlyFace = True, image_size = image_size, 
                          transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[testSplit],
                          isVideo=False, seqLength = 6,dbType = 0)
            else : 
                ID = SEWAFEWReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=listSplit,
                      isVideo=False, seqLength = 6,dbType = 0)
                
                VD = SEWAFEWReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                          transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[testSplit],
                          isVideo=False, seqLength = 6,dbType = 0)
        
        dataloaderTrn = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
        dataloaderVal = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = True)
        
        
        if isSemaine : #-1 to 1
            listV = np.array(range(0,20))
            listA = np.array(range(0,20))
            
            listVx = np.array(range(0,20))
            listAx = np.array(range(0,20))
            
            listVt = np.array(range(0,20))
            listAt = np.array(range(0,20))
            
            
            listVall = np.array(range(0,20))
            listAall = np.array(range(0,20))
            
        elif isAFEW  : #-10 to 10
            listV = np.array(range(0,20))
            listA = np.array(range(0,20))
            
            listVx = np.array(range(0,20))
            listAx = np.array(range(0,20))
            
            listVt = np.array(range(0,20))
            listAt = np.array(range(0,20))
        else : #0 to 1
            listV = np.array(range(0,12))
            listA = np.array(range(0,12))
            
            listVx = np.array(range(0,12))
            listAx = np.array(range(0,12))
            
            listVt = np.array(range(0,12))
            listAt = np.array(range(0,12))
            
        
        
        
        x = 0
        
        temp = []
        
        for real_batch,vas,gt,M,_ in (dataloaderTrn) :
            for va in vas : 
                print(x)
                #print(va,vas)
                
                print(va)
                t = va.cpu().numpy()
                
                if isSemaine : 
                    lva = (va.cpu().numpy()) * 10+10
                    name = 'SE-VA-Train-'
                elif not isAFEW  : 
                    lva = (va.cpu().numpy()) * 10+1
                    name = 'S-VA-Train-'
                    #name = 'SE-VA-Train-'
                else : 
                    #print(va.cpu().numpy())
                    lva = (va.cpu().numpy())+10
                    name = 'VA-Train-'
                    
                listV[int(round(lva[0]))]+=1
                listA[int(round(lva[1]))]+=1
                
                listVall[int(round(lva[1]))]+=1
                listAall[int(round(lva[1]))]+=1
                
                print(lva)
                temp.append(va[0])
                
                x+=1
                
                if minV > t[0]: 
                    minV = t[0]
                if maxV < t[0]: 
                    maxV = t[0]
                if minA > t[1]: 
                    minA = t[1]
                if maxA < t[1]: 
                    maxA = t[1]
        
        '''plt.plot(temp, linestyle=':',marker='s')
        plt.show()'''
        
        x = 0
        
        print(listV,listA)
        np.save(targetDir+name+str(testSplit)+'.npy',np.column_stack((listV,listA)))
        
        for real_batch,vas,gt,M,_ in (dataloaderVal) :
            for va in vas : 
                print(x)
                
                t = va.cpu().numpy()
                
                if isSemaine : 
                    lva = (va.cpu().numpy()) * 10+10
                    #name = 'S-VA-Test-'
                    name = 'SE-VA-Test-'
                elif not isAFEW : #sewa
                    lva = (va.cpu().numpy()) * 10+1
                    name = 'S-VA-Test-'
                else : 
                    lva = (va.cpu().numpy())+10
                    name = 'VA-Test-'
                    
                
                listVt[int(round(lva[0]))]+=1
                listAt[int(round(lva[1]))]+=1
                
                x+=1
                
                if minV > t[0]: 
                    minV = t[0]
                if maxV < t[0]: 
                    maxV = t[0]
                if minA > t[1]: 
                    minA = t[1]
                if maxA < t[1]: 
                    maxA = t[1]
            
        print(listVt,listAt)
        np.save(targetDir+name+str(testSplit)+'.npy',np.column_stack((listVt,listAt)))
        
        
        print('minmax',minA,minV,maxA,maxV)
        
        '''fig, ax = plt.subplots(nrows=1, ncols=2)
        
        for row in ax:
            for col in row:
                col.plot(x, y)'''
        
        fig = plt.figure()
    
        ax = plt.subplot(2, 2, 1)
        ax.bar(listVx,listV)
        ax.set_title('v train')
        
        ax = plt.subplot(2, 2, 2)
        ax.bar(listAx,listA)
        ax.set_title('A train')
        
        ax = plt.subplot(2, 2, 3)
        ax.bar(listVx,listVt)
        ax.set_title('v test')
        
        ax = plt.subplot(2, 2, 4)
        ax.bar(listAx,listAt)
        ax.set_title('A test')
        
        #plt.show()
        plt.savefig(name+'-'+str(split)+".png")
        
        exit(0)
            
    exit(0)
    
    
def checkQuadrant() : 
    
    #Val, arou
    x = [-10,-10]
    y = [-10,10]
    z = [10,-10]
    a = [10,10]
    
    def toQuadrant(inputData = None, min = -10, max = 10,  toOneHot = False):
        threshold = truediv(min,max)
        vLow = False
        aLow = False
        q = 0
        
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
        
    
    print(toQuadrant(inputData = x,toOneHot = True))
    print(toQuadrant(inputData = y,toOneHot = True))
    print(toQuadrant(inputData = z,toOneHot = True))
    print(toQuadrant(inputData = a,toOneHot = True))
