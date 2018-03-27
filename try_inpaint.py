import DAE
import os
import numpy as np
import cv2


dae=DAE.DAE(1)
dae.initModel()
dae.loadModel()

files = os.listdir("inpainting/target/")

for f in files:
    if f.split(".")[-1]=="png":
        print(f)
        img=cv2.imread("inpainting/target/"+f,1)/255.0
        mask=cv2.imread("inpainting/mask/"+f,0)/255.0

        mask=np.reshape(mask,[mask.shape[0],mask.shape[1],1])
        sample=np.append(img,mask,2)
        result=dae.do(sample)

        cv2.imwrite("inpainting/result/"+f,result*255)

dae.close()