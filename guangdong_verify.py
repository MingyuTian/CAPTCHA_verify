#!/usr/bin/env python
#encoding=utf-8
#from numpy import *  
import operator
from os import listdir,remove
from PIL import Image,ImageEnhance,ImageFilter
import sys  
from os.path import isfile, join
import cv2
import numpy as np
from sklearn import preprocessing

class KNN():
    chr1 = np.load("chr1.npy")#training data of the first character for the 4-character case
    lab1 = np.load("lab1.npy")#label of chr1
    chr2 = np.load("chr2.npy")#training data of the second character for the 4-character case
    lab2 = np.load("lab2.npy")#label of chr2
    chr3 = np.load("chr3.npy")#training data of the third character for the 4-character case
    lab3 = np.load("lab3.npy")#label of chr3
    chr4 = np.load("chr4.npy")#training data of the fourth character for the 4-character case
    lab4 = np.load("lab4.npy")#label of chr4
    #change all the label to list
    lab1 = list(unicode(lab1, 'gbk'))
    lab2 = list(unicode(lab2, 'gbk'))
    lab3 = list(unicode(lab3, 'gbk'))
    lab4 = list(unicode(lab4, 'gbk'))
    #the list of all the four-character case
    #all the characters are Chinese
    chengyu_list = ['鹰击长空', '卧薪尝胆', '惊天动地', '万箭穿心', '皆大欢喜', '霸王别姬', '插翅难逃', '偷天换日', '卧虎藏龙', '塞翁失马', '擎天一柱', '郑人买履', '天涯海角', '出生入死', '落叶归根', '长生不死', '绘声绘影', '永无止境', '一见钟情', '花花公子', '金蝉脱壳', '福星高照', '高山流水', '买椟还珠', '兵临城下', '水木清华', '女娲补天', '笑口常开', '精卫填海', '相见恨晚', '醉生梦死', '金玉良缘', '黄道吉日', '相亲相爱', '春暖花开', '风花雪月', '春风化雨', '滔滔不绝', '国士无双', '覆雨翻云', '饮食男女', '韬光养晦', '穿针引线', '珠光宝气', '日日夜夜', '画蛇添足', '愚公移山', '金玉满堂', '六道轮回', '左右逢源', '百里挑一', '亡羊补牢', '妄自菲薄', '一路顺风', '龙生九子', '精忠报国', '八仙过海', '金枝玉叶', '情不自禁', '花好月圆', '相濡以沫', '四海一家', '牛郎织女', '国色天香', '极乐世界', '破釜沉舟', '石破天惊', '养生之道', '黄金时代', '原来如此', '情非得已', '海阔天空', '叶公好龙', '倾国倾城', '否极泰来', '三位一体', '厚德载物', '两小无猜', '世外桃源', '无地自容', '窈窕淑女', '唯我独尊', '掌上明珠', '壮志凌云', '飘飘欲仙', '三皇五帝', '走马观花', '天上人间', '龙马精神', '行尸走肉', '左思右想', '万家灯火', '生财有道', '缘木求鱼', '青梅竹马', '富贵满堂', '背水一战', '随遇而安', '学富五车', '逍遥法外', '海市蜃楼', '作奸犯科', '千军万马', '万里长城', '天下无双', '无忧无虑', '不吐不快', '满腹经纶', '英雄豪杰']
    #unocide
    train1 = [unicode(chengyu_list[i], 'utf8')[0] for i in range(0,len(chengyu_list))]
    train2 = [unicode(chengyu_list[i], 'utf8')[1] for i in range(0,len(chengyu_list))]
    train3 = [unicode(chengyu_list[i], 'utf8')[2] for i in range(0,len(chengyu_list))]
    train4 = [unicode(chengyu_list[i], 'utf8')[3] for i in range(0,len(chengyu_list))]
    judge = np.load("judge.npy")#training data for judging if it is the 4-character case or calculate case
    judgelab = np.load("judgelab.npy")#label for the training data above
    judgelab = list(judgelab)
    pmt = np.load("pmt.npy")#to judge if it is '+', '-' or '*'
    numlab2 = np.load("numlab2.npy")#label for above
    num1 = np.load("number1.npy")#the first digit from 0 to 9
    numlab1 = np.load("numlab1.npy")#label for above
    num2 = np.load("number2.npy")#the second digit from 0 to 9
    numlab3 = np.load("numlab3.npy")#label for above
    numlab2 = list(numlab2)
    numlab1 = list(numlab1)
    numlab3 = list(numlab3)

    #KNN function
    def kNNClassify(self, newInput, dataSet, labels, k): 
       
        numSamples = dataSet.shape[0] # shape[0] stands for the num of row  

        ## step 1: calculate Euclidean distance  
        # tile(A, reps): Construct an array by repeating A reps times  
        # the following copy numSamples rows for dataSet  
        diff = np.tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise  
        squaredDiff = diff ** 2 # squared for the subtract  
        squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row  
        distance = squaredDist ** 0.5  

        ## step 2: sort the distance  
        # argsort() returns the indices that would sort an array in a ascending order  
        sortedDistIndices = np.argsort(distance)  

        classCount = {} # define a dictionary (can be append element)  
        for i in xrange(k):  
            ## step 3: choose the min k distance  
            voteLabel = labels[sortedDistIndices[i]]  

            ## step 4: count the times labels occur  
            # when the key voteLabel is not in dictionary classCount, get()  
            # will return 0  
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  

        ## step 5: the max voted class will return  
        maxCount = 0  
        for key, value in classCount.items():  
            if value > maxCount:  
                maxCount = value  
                maxIndex = key  

        return maxIndex   
      
    #function to cut the image to remove blank spaces from y axis
    #big sum means there are some black cases in this line, so we need to remove the ones of small sum
    #cut the top and bottom white lines and then we can get the lines needed       
    def ycut(self,im):
        col = []
        for i in range(0,len(im)):
            col.append(np.sum(255-im[i]))
        front = [0]
        back = [len(im)-1]
        for i in range(0, 10):
            if col[i] < 3000:
                front.append(i)
            if col[len(im)-1-i] < 3000:
                back.append(len(im)-1-i)
        im1 = im[max(front)+1:min(back)-1,:]
        return im1

    #function to cut the image to remove blank spaces from x axis
    #similiar idea as ycut()
    def xcut(self,im):
        col2 = []
        for i in range(0, im.shape[1]):
            col2.append(np.sum(255-im[:,i]))
        whitelines = []
        for i in range(0,len(col2)):
            if col2[i]> 3000:
                whitelines.append(i)
        im2 = im[:,whitelines]
        return im2


    #opreations for calculation cases
    def split_num(self, f):
        im = Image.open(f)
        #image opeation: remove noises, convert colored images to black-white cases
        im = im.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        im = im.convert('1')
        im.save("a.jpg")
        image_waited = cv2.imread("a.jpg")

        #cut image, resize to make sure every picture is 30*30
        im1 = image_waited[0:35,20:100]
        im2 = self.ycut(im1)
        im3 = self.xcut(im2)
        im4 = cv2.resize(im3, (90,30))
        ima = im4[:,:30]
        imb = im4[:,30:60]
        imc = im4[:,60:90]
        remove("a.jpg")

        return ima[:,:,1].flatten()/255.0, imb[:,:,1].flatten()/255.0, imc[:,:,1].flatten()/255.0

    #knn for calculation, cases like 9 + 1 = ?, but it is in Chinese characters    
    def calres(self, f):
        im1, im2, im3 = self.split_num(f)
        ind = [x*3 for x in range(0,2500)]
        pred1 = self.kNNClassify(im1, self.num1, self.numlab1,1)
        pred2 = self.kNNClassify(im2, self.pmt, self.numlab2,1)
        pred3 = self.kNNClassify(im3, self.num2, self.numlab3,1)
        if pred2 == "m":#m means 'minus'
            return int(pred1) - int(pred3)
        elif pred2 == "t":#t means 'times'
            return int(pred1) * int(pred3)
        elif pred2 == "p":#p means 'plus'
            return int(pred1) + int(pred3)

    #operatoins for 4-character cases
    def split_im(self, f):
        im = Image.open(f)
        im = im.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        im = im.convert('1')
        im.save("a.jpg")

        image_waited = cv2.imread("a.jpg")
        im1 = image_waited[0:35,20:130]
        im2 = cv2.resize(im1, (120,30))
        ima = im2[:,:30]
        imb = im2[:,30:60]
        imc = im2[:,60:90]
        imd = im2[:,90:120]
        remove("a.jpg")

        return ima[:,:,1].flatten()/255.0, imb[:,:,1].flatten()/255.0, imc[:,:,1].flatten()/255.0, imd[:,:,1].flatten()/255.0

    #knn for 4-character cases      
    def chengyu_verify(self, f):
        im1, im2, im3, im4 = self.split_im(f)
        pred1 = self.kNNClassify(im1, self.chr1, self.lab1,1)
        pred2 = self.kNNClassify(im2, self.chr2, self.lab2,1)
        pred3 = self.kNNClassify(im3, self.chr3, self.lab3,1)
        pred4 = self.kNNClassify(im3, self.chr4, self.lab4,1)
        res1 = np.array([int(pred1 == self.train1[i]) for i in range(0,len(self.chengyu_list))])
        res2 = np.array([int(pred2 == self.train2[i]) for i in range(0,len(self.chengyu_list))])
        res3 = np.array([int(pred3 == self.train3[i]) for i in range(0,len(self.chengyu_list))])
        res4 = np.array([int(pred4 == self.train4[i]) for i in range(0,len(self.chengyu_list))])
        #because the 4-character is a fixed, so if we can verify most of the characters right, we can find the case
        res = unicode(self.chengyu_list[np.argmax(res1+res2+res3+res4)], 'utf8')
        
        return res

    #function to judge if it is the calculation case or 4-character cases, based on the fifth character is '于' or nothing    
    def split_judge(self, f):
        im = Image.open(f)
        im = im.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        im = im.convert('1')
        im.save("a.jpg")
        image_waited = cv2.imread("a.jpg")
        im1 = image_waited[0:35,130:150]
        im2 = cv2.resize(im1, (30,30))
        remove("a.jpg")

        return im2[:,:,1].flatten()/255.0
    
    # the main function for verify captcha
    # first judge what case it is, and the veriyf
    def captcha_verify(self, f):
        im= self.split_judge(f)
        pred = self.kNNClassify(im, self.judge, self.judgelab,1)
        if pred == '1':
            return self.calres(f)
        else:
            return self.chengyu_verify(f)
        
