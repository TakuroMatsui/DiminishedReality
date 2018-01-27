import cv2
import numpy as np
import os


class Align:
	def align(self,imageDir,saveDir):
		files = os.listdir(imageDir)
		for f in files:
			img=cv2.imread(imageDir+f,1)
			if not img is None and (f.split(".")[1]=="jpg" or f.split(".")[1]=="JPG" or f.split(".")[1]=="jpeg" or f.split(".")[1]=="JPEG" or f.split(".")[1]=="png"):
				height=img.shape[0]
				width=img.shape[1]

				if height > width:
					sub=height-width
					sub=int(sub/2)
					img=img[sub:sub+width,:]
				if height < width:
					sub=width-height
					sub=int(sub/2)
					img=img[:,sub:sub+height]
				cv2.imwrite(saveDir+f.split(".")[0]+".png",cv2.resize(img,(256,256)))

	def makeMask(self):
		files = os.listdir("data_base/mask/")
		for f in files:
			img=cv2.imread("data_base/mask/"+f,1)
			mask=np.zeros(img.shape,dtype=np.uint8)
			for i in range(img.shape[0]):
				for j in range(img.shape[0]):
					if img[i,j,0]<=10 and img[i,j,1]>=245 and img[i,j,2]<=10:
						mask[i,j]=[255,255,255]
			cv2.imwrite("data_base/mask/"+f,mask)

	def allDo(self):
		print("start")
		self.align("data_source/any_image/","data_base/any_image/")
		self.align("data_source/image/","data_base/image/")
		self.align("data_source/mask/","data_base/mask/")
		self.makeMask()
		print("completed")

if __name__=="__main__":
	print("start")

	al=Align()
	al.allDO()
	print("completed")