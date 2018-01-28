import jbf
import DAE
import Detector
import os
import numpy as np
import cv2

neiborhood8 = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]],np.uint8)

det=Detector.Detector(1)
det.loadModel()
dae=DAE.DAE(1)
dae.loadModel()

files = os.listdir("inpainting/target/")

for f in files:
    print(f)
    img=cv2.imread("inpainting/target/"+f,1)/255.0
    mask=det.do(img)


    mask = cv2.dilate(mask,neiborhood8,iterations=1)
    mask=np.reshape(mask,[mask.shape[0],mask.shape[1],1])

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j,0]>=0.5:
                img[i,j]=[0.0,0.0,0.0]
                mask[i,j,0]=1.0
            else:
                mask[i,j,0]=0.0
    sample=np.append(img,mask,2)

    result=dae.do(sample)

    mask2=1.0-mask[:,:,0]
    jbfResult=np.empty(result.shape,dtype=np.float32)
    for i in range(3):
        indb = img[:,:,i]
        gdb = result[:,:,i]

        w = 30
        sw = 0.05*np.std(np.arange(w))
        cw = np.std(gdb)/2

        jbfResult[:,:,i] = jbf.jbf(gdb,indb,mask2,sw,cw,w)

    cv2.imwrite("inpainting/mask/"+f,mask*255)
    cv2.imwrite("inpainting/result/"+f,jbfResult*255)

det.close()
dae.close()

