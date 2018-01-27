import jbf
import DAE
import Detector
import os
import numpy as np
import cv2

neiborhood8 = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]],np.uint8)

det=Detector.Detector(1)
dae=DAE.DAE(1)

files = os.listdir("inpainting/target/")

for f in files:
    img=cv2.imread("inpainting/target/"+f,1)/255.0
    mask=det.do(img)


    # mask = cv2.dilate(mask,neiborhood8,iterations=1)
    mask=np.reshape(mask,[mask.shape[0],mask.shape[1],1])

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j,0]>=0.5:
                img[i,j]=[0.0,0.0,0.0]
                mask[i,j,0]=1.0
            else:
                mask[i,j,0]=0.0
    sample=np.append(img,mask,2)

    dae.do(sample)

    cv2.imwrite("inpainting/mask/"+f)
    cv2.imwrite("inpainting/result/"+f)

det.close()
dae.close()

