# Improving Robustness of Deepfake Detectors through Gradient Regularization

## Project Overview
This project aims to enhance the robustness of deepfake detection models against adversarial attacks using gradient regularization techniques. As deepfake technology advances, robust detection methods become increasingly critical for security across digital platforms.

## Abstract
Recent generative methods have shown strong capabilities in producing high-quality deepfakes, posing a security threat in various domains, such as social networks and online platforms. Consequently, the demand for deepfake detectors has increased, with an emphasis on high accuracy in identifying deepfakes generated by different methods, as well as ensuring robustness against adversarial attacks to enhance their security. 

Building on that, this project proposes an approach to improve the robustness of deepfake detectors with a gradient regularization technique. This technique has shown promising results in improving model generalization, leveraging an approximation of the Hessian matrix in the gradient calculation to separate the feature space of the trained model by incorporating shallow feature statistics.

## Dataset
[DFFD: Diverse Fake Face Dataset](https://github.com/ondyari/FaceForensics) [3]

*Alternative: FaceForensics++ dataset (requires frame extraction from videos)*

## Methodology
We'll implement the gradient regularization technique described in [1] to improve the adversarial robustness of deepfake detectors. For our base model, we'll use EfficientNetB0 [2], though other suitable architectures may be considered.

## Main Objectives

### 1. Baseline Training & Evaluation
- Train/finetune a deepfake detector without gradient regularization
- Evaluate performance metrics on standard test sets

### 2. Adversarial Attack Assessment
- Apply various adversarial attack methods to evaluate baseline robustness
- Measure performance degradation under attack conditions

### 3. Implement Gradient Regularization
- Train an identical model architecture with gradient regularization
- Evaluate performance on standard test sets

### 4. Robustness Comparison
- Subject the regularized model to the same adversarial attacks
- Compare attack resilience between baseline and regularized models
- Analyze and explain the differences in results

## Expected Outcomes
- Quantitative comparison of detection accuracy between models
- Measurement of robustness against various adversarial attacks
- Analysis of how gradient regularization contributes to improved security

## References
1. W. Guan, W. Wang, J. Dong and B. Peng, (2024). Improving Generalization of Deepfake Detectors by Imposing Gradient Regularization, In IEEE Transactions on Information Forensics and Security, vol. 19, pp. 5345-5356.

2. M. Tan and Q. Le, (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In Proc. Int. Conf. Mach. Learn., pp. 6105–6114.

3. On the Detection of Digital Face Manipulation Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, Anil Jain, (2020), In: Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2020), Seattle, WA, Jun. 2020

4. Abbasi, M., Váz, P., Silva, J. and Martins, P. (2025). Comprehensive Evaluation of Deepfake Detection Models: Accuracy, Generalization, and Resilience to Adversarial Attacks. Applied Sciences, 15(3), 1225.
