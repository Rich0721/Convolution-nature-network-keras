# Convolution-neture-network-keras
Sort out all CNN network can use train and test.
Use keras(2.4.3) and Tensorflow-gpu(2.2.0)

## Todo
- [x] VGGnet
- [x] Resnet
- [x] Xception
- [x] CSPNet
- [x] mobileNet v1
- [x] mobileNet v2

## Install library
```
pip install requirments.txt
```

## Inference
You can construct three folders - train, val, and test, respectively.
And then you want to training categories folder put in these folders.

-train
  -dog
  -cat
-val
  -dog
  -cat
-test
  -dog
  -cat  

```
python train.py
```
```
python test.py
```

## Reference Paper
[VGGNet](https://arxiv.org/abs/1409.1556)
[ResNet](https://arxiv.org/abs/1512.03385)
[MobileNets](https://arxiv.org/abs/1704.04861)
[MobileNets v2](https://arxiv.org/abs/1801.04381)
[CSPNet](https://arxiv.org/abs/1911.11929)
[Xception](https://arxiv.org/abs/1610.02357)
