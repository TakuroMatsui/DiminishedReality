#This is Denoise AutoEncoder with Adversarial Network

import numpy as np
import cv2
import tensorflow as tf
import random
import os
import time
import configparser
from TFLib import TFLib as tfl

class DAE:
    def __init__(self,batchsize=1):
        inifile = configparser.SafeConfigParser()
        inifile.read("settings.ini")
        self.BATCH=batchsize

        self.Size=int(inifile.get("settings","Size"))
        self.Layer=int(inifile.get("settings","Layer"))
        self.Filter=int(inifile.get("settings","Filter"))
        self.Stage=int(inifile.get("settings","Stage"))
        self.Loop=int(inifile.get("settings","Loop"))

        self.directory_name='DAE/data/'
        self.dataBase='data_base/'
        self.datasetDir=self.directory_name+'dataset/'
        self.testsetDir=self.directory_name+'testset/'
        self.modelDir='DAE/model/'
        self.modelname='DAE/model/DAE.ckpt'

        config = tfl().config

        self.graph=tf.Graph()
        self._buidModel()
        self.sess=tf.InteractiveSession(graph=self.graph,config=config)
        self.saver = tf.train.Saver()
        self.testScore=100000.0

        initOP = tf.global_variables_initializer()
        self.sess.run(initOP)


    def loadModel(self):
        self.saver.restore(self.sess, self.modelname)
        fp=open(self.modelDir+"bestLoss.txt","r")
        self.testScore=float(fp.readline())
        fp.close()
    
    def makeDataset(self):
        print("makedataset start")
        files = os.listdir(self.dataBase+"any_image/")
        maskFiles=os.listdir(self.dataBase+"mask/")
        count=0
        for f in files:
            if f.split(".")[-1]=="png":
                print(count)
                img=cv2.imread(self.dataBase+"any_image/"+f,1)
                noisedImg=img.copy()

                while 1:
                    rand=random.randint(0,len(maskFiles)-1)
                    if maskFiles[rand].split(".")[-1]=="png":
                        mask=cv2.imread(self.dataBase+"mask/"+maskFiles[rand],1)
                        break

                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if mask[i,j,0]==255 and mask[i,j,1]==255 and mask[i,j,2]==255:
                            noisedImg[i,j]=[0,0,0]
                            
                if count%10==0:
                    cv2.imwrite(self.testsetDir+'input/'+str(count)+".png",noisedImg)
                    cv2.imwrite(self.testsetDir+'mask/'+str(count)+".png",mask)
                    cv2.imwrite(self.testsetDir+'output/'+str(count)+".png",img)

                else:
                    cv2.imwrite(self.datasetDir+'input/'+str(count)+".png",noisedImg)
                    cv2.imwrite(self.datasetDir+'mask/'+str(count)+".png",mask)
                    cv2.imwrite(self.datasetDir+'output/'+str(count)+".png",img)

                count+=1


    def _readData(self,path,name):
        inputData = cv2.imread(path+'input/'+name,1)/255.0
        outputData=cv2.imread(path+'output/'+name,1)/255.0
        maskData=cv2.imread(path+'mask/'+name,0)/255.0
        maskData=np.reshape(maskData,[-1,self.Size,self.Size,1])[0]

        inputData=np.append(inputData,maskData,2)

        return [inputData],[outputData]

    def _showImages(self,real,fake):
        items=len(real)
        if items>4:
            items=4
        images=np.zeros([self.Size*items,self.Size*2,3])
        for i in range(items):
            images[self.Size*i:self.Size*(i+1),0:self.Size,:]=real[i,:,:,0:3]
            images[self.Size*i:self.Size*(i+1),self.Size:self.Size*2,:]=fake[i,:,:,0:3]
        cv2.imshow("real : fake",images)
        cv2.waitKey(1)
    
    def _buildGenerator(self,x,keep_prob,reuse):
        with tf.variable_scope("Generator") as scope:
            if reuse:
                scope.reuse_variables()

            h=x

            w,b=tfl().conv_variable([self.Filter,self.Filter,4,4],"conv-in1")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,4,8],"conv-in2")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,8,self.Layer],"conv-in3")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            for i in range(self.Stage):

                for j in range(self.Loop):
                    w,b=tfl().conv_variable([self.Filter,self.Filter,self.Layer*(2**i),self.Layer*(2**i)],"conv{0}-{1}".format(i,j))
                    h=tfl().conv2d(h,w,1)+b
                    h=tfl().leakyReLU(h)

                w,b=tfl().conv_variable([self.Filter,self.Filter,self.Layer*(2**i),self.Layer*(2**(i+1))],"conv{0}-out".format(i))
                h=tfl().conv2d(h,w,2)+b
                h=tfl().leakyReLU(h)
                h=tf.nn.dropout(h,keep_prob)
                
            for i in reversed(range(self.Stage)):

                for j in range(self.Loop):
                    w,b=tfl().conv_variable([self.Filter,self.Filter,self.Layer*(2**(i+1)),self.Layer*(2**(i+1))],"deconv{0}-{1}".format(i,j))
                    h=tfl().conv2d(h,w,1)+b
                    h=tfl().leakyReLU(h)

                w,b=tfl().deconv_variable([self.Filter,self.Filter,self.Layer*(2**(i)),self.Layer*(2**(i+1))],"deconv{0}-out".format(i))
                h=tfl().deconv2d(h,w,[self.BATCH,self.Size/(2**i),self.Size/(2**i),self.Layer*(2**i)],2)+b
                h=tfl().leakyReLU(h)
                h=tf.nn.dropout(h,keep_prob)

            w,b=tfl().conv_variable([self.Filter,self.Filter,self.Layer,8],"deconv-out1")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,8,4],"deconv-out2")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,4,3],"deconv-out3")
            h=tfl().conv2d(h,w,1)+b

            y=h

            return y
            
    def _buildDiscriminator(self,x,reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h=x


            w,b=tfl().conv_variable([self.Filter,self.Filter,3,4],"conv-in1")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,4,8],"conv-in2")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,8,self.Layer],"conv-in3")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            for i in range(self.Stage):

                for j in range(self.Loop):
                    w,b=tfl().conv_variable([self.Filter,self.Filter,self.Layer*(2**i),self.Layer*(2**i)],"conv{0}-{1}".format(i,j))
                    h=tfl().conv2d(h,w,1)+b
                    h=tfl().leakyReLU(h)

                w,b=tfl().conv_variable([self.Filter,self.Filter,self.Layer*(2**i),self.Layer*(2**(i+1))],"conv{0}-out".format(i))
                h=tfl().conv2d(h,w,2)+b
                h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,self.Layer*(2**(self.Stage)),8],"conv-out1")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,8,4],"conv-out2")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            w,b=tfl().conv_variable([self.Filter,self.Filter,4,3],"conv-out3")
            h=tfl().conv2d(h,w,1)+b
            h=tfl().leakyReLU(h)

            #fc1
            h=tf.reshape(h,[-1,int(self.Size/(2**self.Stage)*self.Size/(2**self.Stage))*3])
            fc_w1,fc_b1=tfl().fc_variable([self.Size/(2**self.Stage)*self.Size/(2**self.Stage)*3,1],"fc1")
            h=tf.matmul(h,fc_w1)+fc_b1

            y=(tf.nn.tanh(h)+1.0)/2.0

            return y
            


    def _buidModel(self):
        with self.graph.as_default():
            e=0.00000001
            self.gx=tf.placeholder(tf.float32,[None,self.Size,self.Size,4],name="gx")
            self.gy_=tf.placeholder(tf.float32,[None,self.Size,self.Size,3],name="gy_")
            self.learnRate=tf.placeholder(tf.float32)
            self.keep_prob=tf.placeholder(tf.float32)

            self.gy=self._buildGenerator(self.gx,self.keep_prob,False)
            self.g_sample=self._buildGenerator(self.gx,self.keep_prob,True)
            self.g_sample=tf.maximum(0.0,self.g_sample)
            self.g_sample=tf.minimum(1.0,self.g_sample)

            self.dy_real=self._buildDiscriminator(self.gy_,False)
            self.dy_fake=self._buildDiscriminator(self.gy,True)

            self.d_loss_real=tf.reduce_mean(1.0*-tf.log(self.dy_real+e))
            self.d_loss_fake=tf.reduce_mean(1.0*-tf.log(1.0-self.dy_fake+e))
            self.d_loss=self.d_loss_real+self.d_loss_fake

            self.g_loss_fake=tf.reduce_mean(1.0*-tf.log(self.dy_fake+e))*0.0001
            self.g_loss_pix=tf.reduce_mean(tf.abs(self.gy-self.gy_))
            self.g_loss=self.g_loss_pix+self.g_loss_fake

            self.g_optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.g_loss,var_list=[x for x in tf.trainable_variables() if "Generator" in x.name])
            self.d_optimizer = tf.train.AdamOptimizer(self.learnRate/4.0).minimize(self.d_loss,var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])
            
    def train(self,learnRate,keep_prob,TIMES):
        print("train start")
        notUpdate=0
        step=-1

        files = os.listdir(self.datasetDir+'input/')
        filesTest=os.listdir(self.testsetDir+'input/')

        while True:
            step+=1
            for j in range(self.BATCH):
                while 1:
                    rand=random.randint(0,len(files)-1)
                    if files[rand].split(".")[-1]=="png":
                        inputData,outputData=self._readData(self.datasetDir,files[rand])
                        break
                if j==0:
                    input_image=np.array(inputData)
                    output_image=np.array(outputData)
                else:
                    input_image=np.append(input_image,inputData,0)
                    output_image=np.append(output_image,outputData,0)
            
            _,g_loss,g_loss_pix,g_loss_fake,_,d_loss,d_loss_real,d_loss_fake=self.sess.run([self.g_optimizer,self.g_loss,self.g_loss_pix,self.g_loss_fake,self.d_optimizer,self.d_loss,self.d_loss_real,self.d_loss_fake],{self.gx:input_image,self.gy_:output_image,self.learnRate:learnRate,self.keep_prob:keep_prob})

            if step>0 and step % 100 ==0:
                fake=self.sess.run(self.g_sample,{self.gx:input_image,self.keep_prob:1.0})
                self._showImages(input_image,fake)
                print("step : ",step)
                print("g_loss : ",g_loss)
                print("d_loss : ",d_loss)
                print("d_loss_real : ",d_loss_real)
                print("d_loss_fake : ",d_loss_fake)
                print("g_loss_pix : ",g_loss_pix)
                print("g_loss_fake : ",g_loss_fake)


            if step>0 and step % 1000 ==0:
                print(step)
                loss=0.0
                testCount=0.0
                j=0
                testStartTime=time.time()
                for i in range(0,len(filesTest)):
                    if filesTest[i].split(".")[-1]=="png":
                        inputData,outputData=self._readData(self.testsetDir,filesTest[i])
                        if j==0:
                            input_image=np.array(inputData)
                            output_image=np.array(outputData)
                            j+=1
                        else:
                            input_image=np.append(input_image,inputData,0)
                            output_image=np.append(output_image,outputData,0)
                            j+=1
                        if j==self.BATCH:
                            g_loss_pix=self.sess.run(self.g_loss_pix,{self.gx:input_image,self.gy_:output_image,self.keep_prob:1.0})
                            loss+=g_loss_pix
                            testCount=testCount+float(self.BATCH)
                            j=0
                    cv2.waitKey(1)
                loss*=self.BATCH/testCount
                print("learnRate  :"+str(learnRate))
                print("before     : "+str(self.testScore))
                print("current    : "+str(loss))
                fp=open(self.modelDir+"testLog.csv","a")
                fp.write(str(loss)+"\n")
                fp.close()

                print("Test Time:"+str(time.time()-testStartTime)+" s")

                if loss < self.testScore:
                    print("Updated")
                    notUpdate=0
                    self.testScore=loss
                    self.saver.save(self.sess, self.modelname)
                    fp=open(self.modelDir+"bestLoss.txt","w")
                    fp.write(str(loss))
                    fp.close()
                    fp=open(self.modelDir+"LearnRate.txt","w")
                    fp.write(str(learnRate))
                    fp.close()
                else:
                    notUpdate+=1
                    print("NOT Updated: "+str(notUpdate))

            if notUpdate==5:
                learnRate=learnRate*0.1
                notUpdate=0
                self.loadModel()
            
            cv2.waitKey(1)

            if step == TIMES:
                break

#3 channels 0.0~1.0 image and 1 channel mask
    def do(self,sample):
        return self.sess.run(self.g_sample,{self.gx:[sample],self.keep_prob:1.0})[0]

    def close(self):
        with self.sess.as_default():
            self.sess.close()

if __name__=="__main__":
    dae=DAE(1)
    dae.makeDataset()
    dae.close()

    dae=DAE(5)
    # dae.loadModel()
    dae.train(0.0001,0.5,50000) 
    dae.close()