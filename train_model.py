import DAE

dae=DAE.DAE(1)
# dae.loadModel()
dae.train(0.0001,1.0,100000)
dae.close()