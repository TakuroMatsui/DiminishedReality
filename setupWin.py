import os

os.system("mkdir data_base\\any_image data_base\\image data_base\\mask")
os.system("mkdir data_source\\any_image data_source\\image data_source\\mask")


os.system("mkdir DAE\\model")
os.system("mkdir DAE\\data\\dataset\\input DAE\\data\\dataset\\output DAE\\data\\dataset\\mask")
os.system("mkdir DAE\\data\\testset\\input DAE\\data\\testset\\output DAE\\data\\testset\\mask")

os.system("mkdir Detection\\model")
os.system("mkdir Detection\\data\\dataset\\input Detection\\data\\dataset\\output")
os.system("mkdir Detection\\data\\testset\\input Detection\\data\\testset\\output")