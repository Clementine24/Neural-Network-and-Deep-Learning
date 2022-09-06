This paper studies the recognition problem of oracle characters based on few-shot 
learning. For oracle bone data, we restrict access to a large scale of unlabeled
source ancient Chinese characters and a small number of labeled oracle characters.
We propose transfer learning on this task, combining self-supervised learning and
data augmentation. On the basis of transfer learning, we further explored the
Orc-Bert Augmentor based on self-supervised learning pre-training and analyzed
its effect. Specifically, we pre-trained the model using the HWDB dataset and
transferred the parameters of feature extraction from the previous layers to our
classification model. We experimentally demonstrate the effectiveness of our
transfer learning method on this task, compare and analyze the effects of combining
different augmentors. Extensive few-shot learning experiments demonstrate that
our transfer learning method combined with data augmentation greatly improves
the classification accuracy under all network settings in few-shot oracle character
recognition. Our best top-1 accracy under three few-shot setting are 52.9%, 75.8%,
84.3%.
