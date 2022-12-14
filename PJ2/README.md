The first goal of this project is to classify CIFAR-10 dataset. The construction of my network refers to VGG Net and ResNet. I improves the classification accuracy of the network by exploring the structure and the parameters of the neural network. In addition, some techniques are also applied to improve the robustness of the network, such as [label smoothing](https://arxiv.org/abs/1906.02629), [cutmix](https://arxiv.org/abs/1905.04899), [sgdr](https://arxiv.org/abs/1608.03983). And [grad CMA](https://arxiv.org/abs/1610.02391) is used to visualize the area of interest of the model. The classification accuracy of the final model can reach 95.28%.

The second task is to explore the role of batch normalization in the network, verifies the role of BN from the perspectives of loss landscape, gradient predictability, and effect $\beta$- Smoothness. See the link for reference [paper](https://arxiv.org/abs/1805.11604).

The third task is to implement the [DessiLBI optimizer](https://arxiv.org/abs/2007.02010).

See the Report.pdf for specific work and conclusionsã€‚