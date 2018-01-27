#This is Denoise AutoEncoder with Adversarial Network

import numpy as np
import cv2
import tensorflow as tf
import sys
import random
import matplotlib.pyplot as plt
import os
import time

class DAE:
    def __init__(self,batchsize=1):
        self.BATCH=batchsize
        self.size=256
        self.dim=3

        self.Layer=32
        self.Filter=5
        self.Loop=5

        self.directory_name='DAE/data/'
        self.dataBase='data_base/'
        self.datasetDir=self.directory_name+'dataset/'
        self.testsetDir=self.directory_name+'testset/'
        self.modelDir='DAE/model/'
        self.modelname='DAE/model/DAE.ckpt'
        self.logName='DAE/log/DAE.csv'

        self.graph=tf.Graph()
        self._buidModel()
        self.sess=tf.InteractiveSession(graph=self.graph)
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
            print(count)
            img=cv2.imread(self.dataBase+"any_image/"+f,1)
            noisedImg=img.copy()

            rand=random.randint(0,len(maskFiles)-1)
            mask=cv2.imread(self.dataBase+"mask/"+maskFiles[rand],1)

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if mask[i,j,0]==255 and mask[i,j,1]==255 and mask[i,j,2]==255:
                        noisedImg[i,j]=[0,0,0]
                        
            if count%5==0:
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
        maskData=np.reshape(maskData,[-1,self.size,self.size,1])[0]

        inputData=np.append(inputData,maskData,2)

        return [inputData],[outputData]

    def _showImages(self,real,fake):
        items=len(real)
        images=np.zeros([self.size*items,self.size*2,3])
        for i in range(items):
            images[self.size*i:self.size*(i+1),0:self.size,:]=real[i,:,:,0:3]
            images[self.size*i:self.size*(i+1),self.size:self.size*2,:]=fake[i,:,:,0:3]
        images=cv2.resize(images,(256,128*5))
        cv2.imshow("real : fake",images)
        cv2.waitKey(1)

    def _fc_variable(self,weight_shape,name="fc"):
        with tf.variable_scope(name):
            weight_shape=(int(weight_shape[0]),int(weight_shape[1]))
            weight=tf.get_variable("w",weight_shape,initializer=tf.contrib.layers.xavier_initializer())
            bias=tf.get_variable("b",[weight_shape[1]],initializer=tf.constant_initializer(0.1))
        return weight,bias

    def _conv_variable(self,weight_shape,name="conv"):
        with tf.variable_scope(name):
            weight_shape=(int(weight_shape[0]),int(weight_shape[1]),int(weight_shape[2]),int(weight_shape[3]))
            weight = tf.get_variable("w",weight_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias=tf.get_variable("b",[weight_shape[3]],initializer=tf.constant_initializer(0.1))
        return weight,bias

    def _deconv_variable(self,weight_shape,name="deconve"):
        with tf.variable_scope(name):
            weight_shape=(int(weight_shape[0]),int(weight_shape[1]),int(weight_shape[2]),int(weight_shape[3]))
            weight = tf.get_variable("w",weight_shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias=tf.get_variable("b",[weight_shape[2]],initializer=tf.constant_initializer(0.1))
        return weight,bias

    def _conv2d(self,x,w,stride=1):
        return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding="SAME")

    def _deconv2d(self,x,w,output_shape,stride=1):
        output_shape=(int(output_shape[0]),int(output_shape[1]),int(output_shape[2]),int(output_shape[3]))
        return tf.nn.conv2d_transpose(x,w,output_shape=output_shape,strides=[1,stride,stride,1],padding="SAME")

    def _leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x)

    def _maxpool(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    def _buildGenerator(self,x,keep_prob,reuse):
        with tf.variable_scope("Generator") as scope:
            if reuse:
                scope.reuse_variables()
                isTraining=False
            else:
                isTraining=True
            h=x

            #layer 1
            w,b=self._conv_variable([self.Filter,self.Filter,4,self.Layer],"conv1-in")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv1-in-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv1-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv1-norm-{0}".format(i))
                h=self._leakyReLU(h)

            h1=h

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv1-out")
            h=self._conv2d(h,w,2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv1-out-norm")
            h=self._leakyReLU(h)

            h=tf.nn.dropout(h,keep_prob)

            #layer 2
            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv2-in")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv2-in-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv2-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv2-norm-{0}".format(i))
                h=self._leakyReLU(h)

            h2=h

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv2-out")
            h=self._conv2d(h,w,2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv2-out-norm")
            h=self._leakyReLU(h)

            h=tf.nn.dropout(h,keep_prob)

            #layer 3
            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv3-in")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv3-in-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv3-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv3-norm-{0}".format(i))
                h=self._leakyReLU(h)

            h3=h

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv3-out")
            h=self._conv2d(h,w,2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv3-out-norm")
            h=self._leakyReLU(h)

            h=tf.nn.dropout(h,keep_prob)


            #layer 4
            w,b=self._deconv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv1-in")
            h=self._deconv2d(h,w,[self.BATCH,self.size/(2**2),self.size/(2**2),self.Layer],2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv1-in-norm")
            h=self._leakyReLU(h)

            h=tf.concat([h,h3],3)
            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer*2,self.Layer],"U-net1")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="U-net1-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv1-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv1-norm-{0}".format(i))
                h=self._leakyReLU(h)

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv1-out")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv1-out-norm")
            h=self._leakyReLU(h)

            h=tf.nn.dropout(h,keep_prob)

            #layer 5
            w,b=self._deconv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv2-in")
            h=self._deconv2d(h,w,[self.BATCH,self.size/(2**1),self.size/(2**1),self.Layer],2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv2-in-norm")
            h=self._leakyReLU(h)

            h=tf.concat([h,h2],3)
            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer*2,self.Layer],"U-net2")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="U-net2-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv2-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv2-norm-{0}".format(i))
                h=self._leakyReLU(h)

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv2-out")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv2-out-norm")
            h=self._leakyReLU(h)

            h=tf.nn.dropout(h,keep_prob)

            #layer 6
            w,b=self._deconv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv3-in")
            h=self._deconv2d(h,w,[self.BATCH,self.size/(2**0),self.size/(2**0),self.Layer],2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv3-in-norm")
            h=self._leakyReLU(h)

            h=tf.concat([h,h1],3)
            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer*2,self.Layer],"U-net3")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="U-net3-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"deconv3-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="deconv3-norm-{0}".format(i))
                h=self._leakyReLU(h)

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,3],"deconv3-out")
            h=self._conv2d(h,w,1)+b

            y=(tf.nn.tanh(h)+1.0)/2.0

            return y
            
    def _buildDiscriminator(self,x,reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            if reuse:
                scope.reuse_variables()
                isTraining=False
            else:
                isTraining=True

            h=x


            #layer 1
            w,b=self._conv_variable([self.Filter,self.Filter,3,self.Layer],"conv1-in")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv1-in-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv1-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv1-norm-{0}".format(i))
                h=self._leakyReLU(h)

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv1-out")
            h=self._conv2d(h,w,2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv1-out-norm")
            h=self._leakyReLU(h)

            #layer 2
            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv2-in")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv2-in-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv2-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv2-norm-{0}".format(i))
                h=self._leakyReLU(h)

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv2-out")
            h=self._conv2d(h,w,2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv2-out-norm")
            h=self._leakyReLU(h)

            #layer 3
            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv3-in")
            h=self._conv2d(h,w,1)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv3-in-norm")
            h=self._leakyReLU(h)

            for i in range(self.Loop):
                w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv3-{0}".format(i))
                h=self._conv2d(h,w,1)+b
                h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv3-norm-{0}".format(i))
                h=self._leakyReLU(h)

            w,b=self._conv_variable([self.Filter,self.Filter,self.Layer,self.Layer],"conv3-out")
            h=self._conv2d(h,w,2)+b
            h=tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="conv3-out-norm")
            h=self._leakyReLU(h)

            #fc1
            h=tf.reshape(h,[-1,int(self.size/(2**3)*self.size/(2**3))*self.Layer])
            self.fc_w1,self.fc_b1=self._fc_variable([self.size/(2**3)*self.size/(2**3)*self.Layer,1],"fc1")
            h=tf.matmul(h,self.fc_w1)+self.fc_b1

            y=(tf.nn.tanh(h)+1.0)/2.0

            return y
            


    def _buidModel(self):
        with self.graph.as_default():
            e=0.00000001
            self.gx=tf.placeholder(tf.float32,[None,self.size,self.size,4],name="gx")
            self.gy_=tf.placeholder(tf.float32,[None,self.size,self.size,3],name="gy_")
            self.learnRate=tf.placeholder(tf.float32)
            self.keep_prob=tf.placeholder(tf.float32)

            self.gy=self._buildGenerator(self.gx,self.keep_prob,False)
            self.g_sample=self._buildGenerator(self.gx,self.keep_prob,True)

            self.dy_real=self._buildDiscriminator(self.gy_,False)
            self.dy_fake=self._buildDiscriminator(self.gy,True)

            self.d_loss_real=tf.reduce_mean(1.0*-tf.log(self.dy_real+e))
            self.d_loss_fake=tf.reduce_mean(1.0*-tf.log(1.0-self.dy_fake+e))
            self.d_loss=self.d_loss_real+self.d_loss_fake

            self.g_loss_fake=tf.reduce_mean(1.0*-tf.log(self.dy_fake+e))*0.001
            self.g_loss_pix=tf.reduce_mean(-(self.gy_*tf.log(self.gy+e)+(1.0-self.gy_)*tf.log(1.0-self.gy+e)))
            self.g_loss=self.g_loss_pix+self.g_loss_fake

            self.g_optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.g_loss,var_list=[x for x in tf.trainable_variables() if "Generator" in x.name])
            self.d_optimizer = tf.train.AdamOptimizer(self.learnRate/4.0).minimize(self.d_loss,var_list=[x for x in tf.trainable_variables() if "Discriminator" in x.name])
            
    def train(self,learnRate,keep_prob,TIMES):
        print("train start")
        notUpdate=0
        step=-1

        files = os.listdir(self.datasetDir+'input/')
        filesTest=os.listdir(self.testsetDir+'input/')
        testNum=len(filesTest)-self.BATCH-len(filesTest)%self.BATCH

        while True:
            step+=1
            for j in range(self.BATCH):
                rand=random.randint(0,len(files)-1)
                inputData,outputData=self._readData(self.datasetDir,files[rand])
                if j==0:
                    input_image=np.array(inputData)
                    output_image=np.array(outputData)
                else:
                    input_image=np.append(input_image,inputData,0)
                    output_image=np.append(output_image,outputData,0)
            

            if step % 1 ==0:
                _,g_loss,g_loss_pix,g_loss_fake=self.sess.run([self.g_optimizer,self.g_loss,self.g_loss_pix,self.g_loss_fake],{self.gx:input_image,self.gy_:output_image,self.learnRate:learnRate,self.keep_prob:keep_prob})
            if step % 1 ==0:
                _,d_loss,d_loss_real,d_loss_fake=self.sess.run([self.d_optimizer,self.d_loss,self.d_loss_real,self.d_loss_fake],{self.gx:input_image,self.gy_:output_image,self.learnRate:learnRate,self.keep_prob:keep_prob})

            f=open(self.logName,"a")
            f.write(str(g_loss)+","+str(d_loss)+","+str(g_loss_fake)+","+str(d_loss_fake)+"\n")
            f.close()

            if step>0 and step % 100 ==0:
                fake=self.sess.run(self.g_sample,{self.gx:input_image,self.keep_prob:1.0})
                self._showImages(input_image[0:5],fake[0:5])
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
                testStartTime=time.time()
                for i in range(0,len(filesTest)-self.BATCH,self.BATCH):
                    for j in range(self.BATCH):
                        inputData,outputData=self._readData(self.testsetDir,filesTest[i+j])
                        if j==0:
                            input_image=np.array(inputData)
                            output_image=np.array(outputData)
                        else:
                            input_image=np.append(input_image,inputData,0)
                            output_image=np.append(output_image,outputData,0)
                    g_loss_pix=self.sess.run(self.g_loss_pix,{self.gx:input_image,self.gy_:output_image,self.keep_prob:1.0})
                    loss+=g_loss_pix/testNum
                    cv2.waitKey(1)
                loss*=self.BATCH
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

            if notUpdate==10:
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
        self.sess.close()

if __name__=="__main__":
    gan=DAE(1)
    gan.makeDataset()
    gan.close()

    gan=DAE(5)
    # gan.loadModel()
    gan.train(0.0001,0.5,-1) 
    gan.close()