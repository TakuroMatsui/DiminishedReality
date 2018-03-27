import os

os.system("mkdir -p data_base/any_image data_base/mask")
os.system("mkdir -p data_source/any_image data_source/mask")

os.system("mkdir -p DAE/model")
os.system("mkdir -p DAE/data/dataset/input DAE/data/dataset/output DAE/data/dataset/mask")
os.system("mkdir -p DAE/data/testset/input DAE/data/testset/output DAE/data/testset/mask")

os.system("mkdir -p inpainting/target")
os.system("mkdir -p inpainting/result")
os.system("mkdir -p inpainting/mask")