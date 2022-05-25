# Damage Detection System for Wind Turbine Blades using a Siamese Neural Network

An implementation of the [original paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) in pytorch with training and testing on the blade damage dataset. Adapted from: [repository](https://github.com/fangpin/siamese-pytorch).

## Dataset
Clases distribution

| Damage type | Code |
| ----------- | ----------- |
| Leading Edge Erosion | D-2 |
| Lightning Strike Damage | D-3 |
| Crack - Transverse and Longitudinal | D-4/5 |
| No Damage | D-0 |

![alt text](https://github.com/alibarrio/dam-det-WTB/blob/main/images/dam_type_distr_simp.png)

Preprocessing
The images of the damaged regions have different shapes and aspect ratios, so the main preprocessing to be done is to resize them to square images in order to avoid distortions, since the input to the network will be square. To do this, the shortest side of the image is filled with zeros, so that the image is not distorted when resized.
![Example](https://github.com/alibarrio/dam-det-WTB/blob/main/images/d45_res.jpg)

## Training

## Inferences

## Results

