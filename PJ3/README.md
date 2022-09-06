The goal of image captioning is to automatically generate fluent and informative language description of an image for human understanding. As an interdisciplinary task connecting Computer Vision and Nature Language Processing, it explores towards the cutting edge techniques of scene understanding and it is drawing increasing interests in recent years. While recent deep neural network models have achieved promising results on the image captioning task, they rely largely on the availability of corpora with paired image and sentence captions to describe objects in context. We would like to address the task of generating descriptions of novel objects which are not present in paired imagesentence datasets. Novel Image Captioning studies the task of image captioning with novel objects, which only exist in testing images. Intrinsically, this task can reflect the generalization ability of models in understanding and captioning the semantic meanings of visual concepts and objects unseen in training set, sharing the similarity to one/zero-shot learning. The difficulty is that neural network may not work on untrained data. It is a special case of Out-of-Distribution (OOD) generalization problem, which aims to addresses the challenging setting where the testing distribution is unknown and different from the training. For increasing scalability of diversified objects, recently novel object captioning has attracted lots of attention. Most proposed methods are architectural in essence. 

In this project, I try to solve the task of Novel Image Captioning. Given an input scene image, the objective is to the corresponding informative language description. All the three standard automatic evaluations metrics: CIDEr-D [9], METEOR [2] and SPICE[1] were used to evaluate the performance of captions. F1 scores should also be reported for evaluating the performance of captioning eight novel concepts.

Datasets: the official [MSCOCO](https://cocodataset.org/#download) dataset

The annotation and split: [Google Link](https://drive.google.com/drive/u/0/folders/1ct0KhDW8ZHW4D9pxu0IX1ntTaH-XOAVV)

Preparation for dataset: refer to this [website](https://github.com/LisaAnne/DCC)

This project is according to this [Neural Baby Talk](https://arxiv.org/abs/1803.09845).

See the Report.pdf for specific work and conclusions.