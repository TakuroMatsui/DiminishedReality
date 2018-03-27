import align_image
import DAE

dataSetup=align_image.Align()
dataSetup.allDo()

dae=DAE.DAE(1)
dae.initModel()
dae.makeDataset()
dae.close()