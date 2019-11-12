# Monte_carlo_dropout
Using monte carlo dropout to have an estimation of predictions uncertainty

### instalation
```
cd monte_carlo_dropout
pip install -e ./
```

###usage
executing the ` unet_learner` function will give you the modified unet with dropout.
using the `DropOutAlexnet` class will give you the alexnet architecture with dropout added.

### credits:
__Fastai online resources:__ 
- Complete code for image segmentation with the One Hundred Layers Tiramisu (FC-DenseNet model)    https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid-tiramisu.ipynb 
- Research papers used
  - Dropout as a Bayesian Approximation: Representing Model: https://arxiv.org/abs/1506.02142
  - Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference:  https://arxiv.org/abs/1506.02158 
  - Nature paper : https://www.nature.com/articles/s41598-019-50587-1?fbclid=IwAR3vS2Jsa16NtOdFgp-I_deIwT8ipsK0isY6oIzBeaPHjOllhDSv1FfAVGg


