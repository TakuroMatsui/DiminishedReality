import numpy as np
import cv2

def jbf(g,d,mask,sw,cw,w): 
	x = np.arange(-w,w+1)
	y = np.arange(-w,w+1)
	[mesh1,mesh2] = np.meshgrid(x,y)
	sp = np.exp(-(mesh1*mesh1+mesh2*mesh2)/(2*sw*sw))

	r = np.zeros(d.shape,dtype=np.float64)
	c = np.zeros(d.shape,dtype=np.float64)
	hi = g.shape[0]
	we = g.shape[1]

	for i in np.arange(hi):
		for j in np.arange(we):

			wimi = np.maximum(i-w,0)
			wima = np.minimum(i+w+1,hi)
			wjmi = np.maximum(j-w,0)
			wjma = np.minimum(j+w+1,we)

			ilocal = g[wimi:wima,wjmi:wjma]

			dlocal = d[wimi:wima,wjmi:wjma]

			mlocal = mask[wimi:wima,wjmi:wjma]
		#	lm = dlocal>10.0

			di = (ilocal - g[i,j]) 
			iw = np.exp(-di**2/(cw**2))* mlocal
			dw = iw * sp[wimi-i+w:wima-i+w,wjmi-j+w:wjma-j+w]

			c[i,j] = np.sum(dw)
			r[i,j] = np.sum(dw*dlocal)/c[i,j]

	return r

		

		

if __name__ == "__main__":


	#emsk = np.zeros(gt.shape,dtype = np.bool_)
	#emsk = np.isnan(gt)

	#gt[emsk]=0

	outfiledir = 1
	ind = cv2.imread('sample/3rdgroup/raw.png')/255.0
	gd = cv2.imread('sample/3rdgroup/gd.png')/255.0
	mask =1-(cv2.cvtColor(cv2.imread('sample/3rdgroup/mask.png'), cv2.COLOR_BGR2GRAY)/255.0)
	res = np.empty(ind.shape,dtype = np.float64)
	for c in [0,1,2]:
		indb = ind[:,:,c]
		gdb = gd[:,:,c]


		w = 50
		sw = 0.05*np.std(np.arange(w))
		cw = np.std(gdb)/2

		res[:,:,c] = jbf(gdb,indb,mask,sw,cw,w) 
	#	norm = np.amax(ind)
	#	print norm
	cv2.imshow("iw",res)
	cv2.waitKey(0)
	cv2.imwrite("jbf.png",res*255)

		#b = jbf(grey,a,10,np.std(grey),15)
