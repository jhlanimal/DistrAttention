# DistrAttentin framework
## Requirements
DistrAttentin requires an installed Python3 version to run the scripts. The version used to develop the project is Python3.11.7. For convenience, we provide a requirements.txt file listing all the packages needed to run the scripts, which can all be installed at once by running:
```
pip install -r requirements.txt 
```
## Deployment
In all the experiments, the sorting algorithm in the Triton library needs to be modified to return indices instead of the default values.

In the ViT experiment, during runtime, the ```models``` folder under each corresponding method in the ViT directory needs to be replaced with the ```models``` folder from the ```timm``` library in the Python environment.

In both LLaMA and BERT, the attention module needs to be replaced with the DistrAttention module.

## Dataset
ImageNet(https://image-net.org/index.php)\
CIFAR-100(https://www.cs.toronto.edu/~kriz/cifar.html)\
CIFAR-10(https://www.cs.toronto.edu/~kriz/cifar.html)\
iNaturalist 2018(https://github.com/visipedia/inat_comp/tree/master/2018)\
iNaturalist 2019(https://github.com/visipedia/inat_comp/tree/master/2019)\
MathInstruct(https://huggingface.co/datasets/TIGER-Lab/MathInstruct/tree/main)

