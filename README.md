# Inception-v1 for Image Classification

- TensorFlow implementation of [Going Deeper with Convolutions](https://research.google.com/pubs/pub43022.html). 
- Training a Inception V1 network from scratch on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Requirements
- Python 3.6
- TensorFlow 1.12+

## ImageNet Classification

### Image classification using pre-trained model
Download the pre-trained parameters [here](http://www.deeplearningmodel.net/).

|Data Source | Image | Result |
|:--|:--:|:--|
|[COCO](http://cocodataset.org/#home) |<img src='data/coco/000000000064.jpg' height='200px'>|1: probability: 0.35, label: analog clock<br>2: probability: 0.29, label: wall clock<br>3: probability: 0.15, label: scale, weighing machine<br>4: probability: 0.11, label: barometer<br>5: probability: 0.04, label: sundial<br>|
|[COCO](http://cocodataset.org/#home) |<img src='data/coco/000000000074.jpg' height='200px'>|1: probability: 0.38, label: Labrador retriever<br>2: probability: 0.05, label: kuvasz<br>3: probability: 0.04, label: Great Pyrenees<br>4: probability: 0.04, label: bloodhound, sleuthhound<br>5: probability: 0.02, label: golden retriever<br>|
|[COCO](http://cocodataset.org/#home) |<img src='data/coco/000000000133.jpg' height='200px'>|1: probability: 0.28, label: file, file cabinet, filing cabinet<br>2: probability: 0.25, label: chiffonier, commode<br>3: probability: 0.18, label: desk<br>4: probability: 0.06, label: chest<br>5: probability: 0.03, label: bookcase<br>|
|[voc2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) |<img src='data/voc2007/000015.jpg' height='200px'>|1: probability: 0.39, label: tricycle, trike, velocipede<br>2: probability: 0.10, label: bicycle-built-for-two, tandem bicycle, tandem<br>3: probability: 0.09, label: moped<br>4: probability: 0.09, label: jinrikisha, ricksha, rickshaw<br>5: probability: 0.04, label: mountain bike, all-terrain bike, off-roader<br>|
|[voc2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) |<img src='data/voc2007/000031.jpg' height='200px'>|1: probability: 0.66, label: electric locomotive<br>2: probability: 0.34, label: passenger car, coach, carriage<br>3: probability: 0.00, label: streetcar, tram, tramcar, trolley, trolley car<br>4: probability: 0.00, label: bullet train, bullet<br>5: probability: 0.00, label: freight car<br>|
|[voc2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) |<img src='data/voc2007/000057.jpg' height='200px'>|1: probability: 1.00, label: hornbill<br>2: probability: 0.00, label: toucan<br>3: probability: 0.00, label: vulture<br>4: probability: 0.00, label: black stork, Ciconia nigra<br>5: probability: 0.00, label: bald eagle, American eagle, Haliaeetus leucocephalus<br>|
|[HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html) |<img src='data/HKU-IS/0008.png' height='200px'>|1: probability: 0.42, label: African elephant, Loxodonta africana<br>2: probability: 0.34, label: tusker<br>3: probability: 0.14, label: Indian elephant, Elephas maximus<br>4: probability: 0.07, label: water buffalo, water ox, Asiatic buffalo, Bubalus bubalis<br>5: probability: 0.01, label: warthog<br>|
|[HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html) |<img src='data/HKU-IS/0015.png' height='200px'>|1: probability: 0.61, label: impala, Aepyceros melampus<br>2: probability: 0.11, label: hartebeest<br>3: probability: 0.09, label: gazelle<br>4: probability: 0.07, label: dhole, Cuon alpinus<br>5: probability: 0.06, label: bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis<br>|
|[HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html) |<img src='data/HKU-IS/0595.png' height='200px'>|1: probability: 0.95, label: horse cart, horse-cart<br>2: probability: 0.02, label: plow, plough<br>3: probability: 0.01, label: oxcart<br>4: probability: 0.01, label: ox<br>5: probability: 0.01, label: cannon<br>|





## Train the network on CIFAR-10

#### Preparation
- Download CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
- Setup path in [`tools/inception_cifar.py`](tools/inception_cifar.py): 
`DATA_PATH` is the path to put CIFAR-10. 
`SAVE_PATH` is the path to save or load summary file and trained model.
#### Train the model
Go to `tools/` and run the script:

```
python inception_cifar.py --train \
  --lr LEARNING_RATE \
  --bsize BATCH_SIZE \
  --keep_prob KEEP_PROB_OF_DROPOUT \
  --maxepoch MAX_TRAINING_EPOCH
```

#### Evaluate the model
Go to `tools/` and put the pre-trained model in `SAVE_PATH`. Then run the script:

```
python inception_cifar.py --eval \
```
learning curve for training set 

![train_lc](imgs/train.png)

learning curve for testing set 
- The accuracy on testing set is 0.9359.

![valid_lc](imgs/valid.png)

