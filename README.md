# CIFAR10 with Pytorch and small GPU

I am trying to get the best results with a small GPU on the CIFAR10 dataset.

## Getting Started

The results are given with my 8Go GPU. You can run these examples, with less Go, if you split mini-batches in micro-batches with gradient accumulation technic. However, the results are slightly different. I guess this is because the data transformations are not the same. 

### Prerequisites

* Python 3.6+

* PyTorch 1.0+

* [EfficientNet-Pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) - Install with  ```pip install efficientnet_pytorch```.

### Usage and expected accuracy

* For a single model with 8Go GPU: ```python main.py```.

  test accuracy : 98.3%

* With a 2Go GPU : python main.py --batch_split 4

* You can use *ensemble estimation* to improve the test accuracy. For example,  to do 10 estimates with 8Go GPU: ```python main10.py```. After the learning, the following script will compute the mean prediction of the models on test dataset: ```python aggregate.py```.

  test accuracy : 98.5%.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Some explanations

* I used transfer learning with the EfficientNet-B4 pretrained model.

* I used [Mixup](https://github.com/facebookresearch/mixup-cifar10), a generic and straightforward data augmentation principle.


## Acknowledgement
The *utils.py* file is from the [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) repository by [kuangliu](https://github.com/kuangliu).
