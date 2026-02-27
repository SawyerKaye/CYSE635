# Introduction

Adversarial images/attacks in AI can have a wide array of effects, from simply causing wrong output from a chatbot to causing a person to crash in a self-driving car that mistook a stop sign for something else. There is therefore a need to understand how these attacks can be implemented, largely in effore to be able to guard against them. In this lab I utilized two different types of adversarial attacks on a CNN model to degrade its performance.

# Methodology

## CNN Architecture

I used the same CNN architecture as my EfficientNet from our CNN Project, which includes two convolutional layers separated by a batch normalization and Rectified Linear Unit layer, which are then followed by a max pooling layer and finally three fully connected layers. My model was trained using the same EarlyStop function as my previous project, and was overall trained on ~4 epochs (due to early stopping) with >98% accuracy when evaluated on the testing data.

## Adversarial method: Fast Gradient Sign Method (FGSM)

I implemented an FGSM method based on the example code given in the course with a step size of .2, which resulted in my model's accuracy dropping to 5% when evaluated on the altered images, though when displayed the images showed clear sign of noise. For clarity, an FGSM is a method in which the entire image is slightly altered based on the sign of the gradient resulting from model evaluation. This results in a doctored image that, depending on the step size, may not show any sign of alteration. After implementing the FGSM, I attempted to tune the step size parameter to its optimum level, employing 5 different step sizes and plotting their accuracies and resulting images, which I shall discuss in the results and analysis sections.

## Adversarial method: Projected Gradient Descent (PGD)

I implemented a PGD method based on the example code given in the course with an epsilon of .3, a step size of .2, and 40 iterations. This resulted in my model's accuracy dropping to 0, though the images show clear signs of tampering. A PGD method is implemented simlarly to an FGSM, but involves using the gradient sign with the step size over multiple iterations where each iteration is applied on top of the old one. After implementing the PGD, I first attempted to tune the iterations number to try to maintain low accuracy while removing some of the noise, however found that my results were largely 0% accuracy with clear signs of doctoring. I instead tuned the epsilon parameter, which is the paramaeter that determines the maximum amount of change in an image. I will discuss the results of my 5 different epsilon values in the results and analysis sections.

# Results

When testing the model on the original images, the FGSM images, and the PGD images, I achieved the below results:

IMAGE

As shown above, both attacks caused model performance to drop, with PGD outperforming FGSM in reducing accuracy, though both were definitely sufficiently powerful.

When comparing accuracies across tuned parameters, FGSM performance is shown below for the given step size

PGD performance is shown below for the given epsilon value

# Analysis

## Basic Image Analysis

Below you can see the affect that the different models had on the images:

As clearly can be seen, the adversarial images show clear signs of alteration, which would explain why the model performed so poorly on them. I additionally developed feature maps from the model for all of the images, shown below:

Original

FGSM

PGD

As can be seen above, the model could fairly clearly make out the original images, however with FGSM and even more so PGD, the added noise attack caused the image to become obfuscated, shedding further light on why the model had such poor performance.

## FGSM Parameter Tuning Analysis

Below is shown the images from the five different step levels for the FGSM method

Based on the above, once the step size increases beyond .15, the image shows clear signs of alteration (though a discerning eye may also notice it at the .15 step size). Given the accuracy at .15 is still only ~12%, if this method were to truly be implemented, I would recommend utilizing that step size as it does the best job of balancing the covertness and effectiveness of the attack.

## PGD Parameter Tuning Analysis

Below is shown the images from the five different epsilon values for the PGD method

Based on the above, although epsilon values >.15 result in 0% accuracy, they also result in clear signs of tampering, indicating that an epsilon value of .15 is ideal as the attack may go unnoticed and the accuracy drops to <2%

# Conclusion

Both FGSM and PGD are effective methods for impacting model accuracy, though their implementation in this lab often resulted in clear signs of tampering and required parameter tuning to mitigate this flaw. Based on the accuracy loss and difficulty in detection, I would recommend someone utilize PGD with an epsilon of .15 if computing is not an issue, or FGSM with a step size of .15 if they are looking for a quicker attack.

In terms of countermeasures/improvements for the model and previous steps to protect against attack, I recommend the following: 

1) Develop a more robust dataset - including images made by PGD and FGSM would improve the model's readiness to handle these issues
2) Utilize randomness factors like Dropout layers within the model itself that would further improve its readiness for noise, as well as possibly help mitigate some noise from the attacks
3) Implement methods to check images for anomalies - having a function that could possibly detect anomalies could prevent the adversary's ability to impact the system, and doing so in an auditable manner would allow someone to finetune the screening function to minimize impact to legitimate input.
4) Rescale the image - Change the scale of the image down and then back to the original, as a lot of the noise was in the void spaces, and the rescaling may naturally smooth over a lot of the rough areas.

Overall this lab was incredibly interesting and it was a lot of fun to get hands on experience in implementing these methods. Additionally, by implementing and evaluating them, I feel better prepared to potentially guard against them.

My code can be found here: 

Sources:

Course Example Code
Feature Map Visualization - https://www.geeksforgeeks.org/deep-learning/visualizing-feature-maps-using-pytorch/?utm_source=chatgpt.com
