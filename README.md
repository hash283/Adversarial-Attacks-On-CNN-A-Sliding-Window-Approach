# Adversarial-Attacks-On-CNN-A-Sliding-Window-Approach
This repository contains the Tensorflow implementation of a sliding window version of the popular FGSM adversarial attack on various CNN models trained on the Facescrub dataset

The Fast Gradient Sign Method(FGSM) - [Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014b.](https://arxiv.org/abs/1412.6572) attack is a popular and effective attack technique used in the field of adversarial machine learning. The primary goal of the FGSM attack is to perturb the input data in a way that a machine learning model misclassifies the modified data while keeping the perturbations imperceptible to human observers.

The FGSM attack works by taking advantage of the model's gradients with respect to the input data. These gradients indicate how sensitive the model's predictions are to changes in the input features. The attack starts with a clean input data point, and the model's gradients are calculated with respect to the loss function that measures the model's performance. Then, the input data is perturbed in the direction that maximizes the loss by adding or subtracting a small value (usually scaled by a hyperparameter known as the epsilon value) to each feature of the input. The attack is efficient and computationally inexpensive because it requires only one forward and backward pass through the model to calculate the gradients and create the adversarial example.

FGSM is widely used to evaluate the robustness of machine learning models, understand their vulnerabilities to adversarial attacks, and develop defense mechanisms to enhance model security. 

Here an iterative sliding window version of this attack is used to perform attacks on the model. The advantage of this method over the conventional method is that it is an iterative method over the traditional one-shot FGSM attack. Moreover, the sliding window technique can be used to find the regions of the image which are most susceptible to perturbations that causes misclassifications in the model.

Various models such as VGG-16, Resnet, etc pretrained on the [FACESCRUB Dataset](https://vintage.winklerbros.net/facescrub.html) are used as models on which these attacks are performed.

## Usage

- Run Dataset1_cuda.py to download the FACESCRUB dataset.
- Run final_vgg.py to perform an attack on the VGG-16 model trained on the FACESCRUB Dataset.

## Results
<img src="./Images/New folder/1.png">
<img src="./Images/New folder/2.png">
<img src="./Images/New folder/3.png">
<img src="./Images/New folder/4.png">
<img src="./Images/New folder/5.png">
<img src="./Images/New folder/6.png">
<img src="./Images/New folder/7.png">
<img src="./Images/New folder/8.png">


The above results of the sliding window FGSM attack shows the most prominent regions of the face which can cause model misclassification if these areas are perturbed in the correct manner. Particularly in the FACESCRUB Dataset, the eyes and the nose are the most prone areas to attack for model misclassification.
