import align_image
import DAE
import Detector

dataSetup=align_image.Align()

det=Detector.Detector(1)
det.makeDataset()
det.close()

dae=DAE.DAE(1)
dae.makeDataset()
dae.close()

det=Detector.Detector(5)
det.train(0.0001,0.5,10000) 
det.close()

dae=DAE.DAE(5)
dae.train(0.0001,0.5,100000) 
dae.close()