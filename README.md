# Image-to-Image-Translation

## Some applications of this implementation

<img width="468" alt="image" src="https://github.com/rraghavkaushik/Image-to-Image-Translation/assets/136466980/04726e00-ec34-4faf-ac45-6ea1a2e0b193">

This repo is an implementation of the paper "Image-to-Image Translation with Conditional Adversarial Networks" 

## Some key points from the paper 

- This paper uses Conditional GANs which is very similar to traditional GANs except that it has a class label for every train image
- For the generator, a U-Net Based architechture is used
- For the discriminator, we use a PatchGAN classifier
- We use minibatch Stochastic Gradient Desent(SGD) and apply Adam solver with a learning rate of 0.0002 and momentum parameters B1 = 0.5 and B2 = 0.999
- 


## Generator

<img width="317" alt="image" src="https://github.com/user-attachments/assets/6d32921c-88f9-4f0e-a997-382b636b494b">
(🙃Image is too big to take a good ss, so refer to the architechture)


## Discriminator

<!--- <img width="317" alt="image" src="https://github.com/user-attachments/assets/bcb9fbfb-0bb4-4239-a120-a061e8e99653"> --->
<img width="317" alt="image" src="https://github.com/user-attachments/assets/7d3f5079-12d7-4f78-a3e4-a25ea4534878">
<img width="809" alt="image" src="https://github.com/user-attachments/assets/ac40a4a7-46c5-42c2-b9cd-256d6ab4ec76">



Link to the paper "https://arxiv.org/pdf/1611.07004"
