import DAE
import random
import configparser

score=1000000.0
inifile = configparser.SafeConfigParser()
inifile.read("settings.ini")

while 1:
    dae=DAE.DAE(10)
    dae.Layer=random.randint(8,16)
    while 1:
        dae.Filter=random.randint(3,5)
        if dae.Filter % 2 !=0:
            break
    dae.Stage=random.randint(0,2)
    dae.Loop=random.randint(0,5)
    dae.initModel()
    print(dae.Layer)
    print(dae.Filter)
    print(dae.Stage)
    print(dae.Loop)
    dae.train(0.0001,0.5,30000)
    dae.close()
    if score > dae.testScore:
        score=dae.testScore
        f=open("searchResult.ini","w")
        f.write("[settings]\n")
        f.write("Size="+str(int(inifile.get("settings","Size")))+"\n")
        f.write("Layer="+str(dae.Layer)+"\n")
        f.write("Filter="+str(dae.Filter)+"\n")
        f.write("Stage="+str(dae.Stage)+"\n")
        f.write("Loop="+str(dae.Loop)+"\n")
        f.close()
