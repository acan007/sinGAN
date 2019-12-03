# sinGAN Pytorch Implementation

## Paper

[SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/abs/1905.01164), ICCV 2019, Best Paper Award


## Getting Started

### Prerequisite
 

### Installation
- Clone this repo:
```bash
git clone https://github.com/bocharm/sinGAN.git
cd sinGAN
```
- Install PyTorch and dependencies from http://pytorch.org   

### Model Training
- Train a model - basic random sample:
```
python main.py
```

- Train a model - paint2image, editing, harmonization ..:
```
python main.py --config ./config/paint2image.yaml
```

- Train a model - super_resolution:
```
python main.py --config ./config/SR.yaml
```

- Test a model:
```
python applications.py --config ./config/editing.yaml --mode editing
```


## Results of this implementation

#### Random samples
- Train Image

![](assets/Input/Images/birds.png)
- Model Output (Random Sampled)

![](assets/samples/birds_randomsample.jpg)

#### paint to image
- Train Image 

![](assets/Input/Images/cows.png)

- Naive Image

![](assets/Input/Paint/cows.png)

- Model Output (image order : from coarsest scale to finest scale)

![](assets/samples/cows_paint2image.png)

#### Harmonization
- Train Image 

![](assets/Input/Images/starry_night.png)

- Naive Image

![](assets/Input/Harmonization/starry_night_naive.png)

- Model Output (image order : from coarsest scale to finest scale)

![](assets/samples/starry_night_harmonization.png)

#### Editing
- Train Image 

![](assets/Input/Images/stone.png)

- Naive Image

![](assets/Input/Editing/stone_edit.png)

- Model Output (image order : from coarsest scale to finest scale)

![](assets/samples/stone_editing.png)


## TODO
- Other task in paper - Paint to img, Editing, Harmonization, SR, Animation
- Arbitrary aspect ratio, resolution

## Reference 
[sinGAN](https://github.com/tamarott/SinGAN) (Author Implementation)
