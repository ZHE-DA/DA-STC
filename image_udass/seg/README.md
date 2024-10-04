# Unified Domain Adaptive Semantic Segmentation

## Main Results

#### GTAV → CITYSCAPES

| Methods                | road           | side.          | buil.          | wall           | fence          | pole           | light          | sign           | vege.          | terr.          | sky            | per.           | rider          | car            | truck          | bus            | train          | moto.          | bike           | mIoU           |
| ---------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Source                 | 63.3           | 15.7           | 59.4           | 8.6            | 15.2           | 18.3           | 26.9           | 15.0           | 80.5           | 15.3           | 73.0           | 51.0           | 17.7           | 59.7           | 28.2           | 33.1           | 3.5            | 23.2           | 16.7           | 32.9           |
| Sepico(CNN)            | 95.2           | 67.8           | 88.7           | 41.4           | 38.4           | 43.4           | 55.5           | 63.2           | 88.6           | 46.4           | 88.3           | 73.1           | 49.0           | 91.4           | 63.2           | 60.4           | 0.0            | 45.2           | 60.0           | 61.0           |
| Freedom(CNN)           | 90.9           | 54.1           | 87.8           | 44.1           | 32.6           | 45.2           | 51.4           | 57.1           | 88.6           | 42.6           | 89.5           | 68.8           | 40.0           | 89.7           | 58.4           | 62.6           | 55.3           | 47.7           | 40.0           | 61.3           |
| **QuadMix(CNN)** | 97.1           | 78.0           | 90.4           | 49.7           | 40.3           | 53.4           | 61.7           | 70.9           | 90.7           | 49.7           | 92.9           | 77.9           | 53.9           | 93.5           | 72.4           | 65.9           | 0.7            | 60.5           | 68.5           | 66.8           |
| Sepico(ViT)            | 96.7           | 76.7           | 89.7           | 55.5           | 49.5           | 53.2           | 60.0           | 64.5           | 90.2           | 50.3           | 90.8           | 74.5           | 44.2           | 93.3           | 77.0           | 79.5           | 63.6           | 61.0           | 65.3           | 70.3           |
| Freedom(ViT)           | 96.7           | 74.8           | 90.9           | 58.1           | 49.0           | 57.5           | 63.4           | **71.4** | 91.6           | 52.1           | **94.4** | 78.4           | 53.1           | 94.1           | 83.9           | 85.2           | 72.5           | 62.8           | **68.9** | 73.6           |
| **QuadMix(ViT)** | **97.5** | **80.9** | **91.6** | **62.3** | **57.6** | **58.2** | **64.5** | 71.2           | **91.7** | **52.3** | 94.3           | **80.0** | **55.9** | **94.6** | **86.3** | **90.5** | **82.3** | **65.1** | 68.1           | **76.1** |

#### SYNTHIA → CITYSCAPES

|        Methods        | road           | side.          | buil.          | wall*          | fence*         | pole*          | light          | sign           | vege.          | sky            | per.           | rider          | car            |      bus      |     motor     |      bike      |    mIoU(16)    |    mIoU(13)    |
| :--------------------: | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | :------------: | :------------: | :------------: | :------------: | :------------: |
|         Source         | 36.3           | 14.6           | 68.8           | 9.2            | 0.2            | 24.4           | 5.6            | 9.7            | 69.0           | 79.4           | 52.5           | 11.3           | 49.8           |      9.5      |      11.0      |      20.7      |      33.7      |      29.5      |
|      Sepico(CNN)      | 77.0           | 35.3           | 85.1           | 23.9           | 3.4            | 38.0           | 51.0           | 55.1           | 85.6           | 80.5           | 73.5           | 46.3           | 87.6           |      69.7      |      50.9      |      66.5      |      58.1      |      66.5      |
|      Freedom(CNN)      | 86.0           | 46.3           | 87.0           | 33.3           | 5.3            | 48.7           | 38.1           | 46.8           | 87.1           | 59.1           | 71.2           | 38.1           | 87.1           |      54.6      |      51.3      |      59.9      |      59.1      |      66.0      |
| **QuadMix(CNN)** | 88.5           | 52.9           | 87.1           | 3.7            | 1.4            | 56.2           | 62.7           | 59.2           | 87.2           | 89.0           | 79.1           | 55.8           | 87.9           |      61.7      |      58.1      |      71.2      |      60.9      |      67.4      |
|      Sepico(ViT)      | 87.0           | **52.6** | 88.5           | 40.6           | **10.6** | 49.8           | 57.0           | 55.4           | 56.8           | 86.2           | 75.4           | 52.7           | **92.4** |      78.9      |      53.0      |      62.6      |      64.3      |      71.4      |
|      Freedom(ViT)      | **89.4** | 50.8           | **89.3** | **48.8** | 9.3            | 57.3           | **65.1** | 60.1           | **89.9** | 93.7           | 79.4           | 51.6           | 90.5           |      66.0      |      62.3      | **68.1** |      67.0      |      73.6      |
| **QuadMix(ViT)** | 88.1           | 51.2           | 88.9           | 46.7           | 7.9            | **58.6** | 64.7           | **63.7** | 88.1           | **93.9** | **81.3** | **56.6** | 90.3           | **66.9** | **66.8** |      66.0      | **67.5** | **74.3** |

## Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/udass-seg
source ~/venv/udass-seg/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

## Dataset Setup

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia:** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

The final folder structure should look like this:

```none
VIDEO
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Training

For the experiments in our paper, we use a script to automatically
generate and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

(The core code will be available soon.)

## Evaluation

A trained model can be evaluated using:

```shell
sh test.sh work_dirs
```

The checkpoints should be downloaded and be put in ./work_dirs.

The predictions are saved for inspection to
`work_dirs/preds`
and the mIoU of the model is printed to the console.

When training a model on Synthia→Cityscapes, please note that the
evaluation script calculates the mIoU for all 19 Cityscapes classes. However,
Synthia contains only labels for 16 of these classes. Therefore, it is a common
practice in UDA to report the mIoU for Synthia→Cityscapes only on these 16
classes. As the Iou for the 3 missing classes is 0, you can do the conversion
`mIoU16 = mIoU19 * 19 / 16`.

The predictions can be submitted to the public evaluation server of the
respective dataset to obtain the test score.

(The evalution code can in parallel with Train) (run the evalution code to validate the results.)

## Checkpoints

Below, we provide checkpoints of UDASS(HRDA) for the different benchmarks.

* [image-UDASS(VIT) for GTA→Cityscapes](https://drive.google.com/file/d/1VYlG0f92Y8VAv712-i5f4GEIwTqbURqf/view?usp=sharing)
* [image-UDASS(VIT) for Synthia→Cityscapes](https://drive.google.com/file/d/1ll6BAqoexkNDOpLJ3eaJRIBgkysnMwsZ/view?usp=sharing)
* [image-UDASS(CNN) for GTA→Cityscapes](https://drive.google.com/file/d/15ryaQVPAcwuvx42N4ag-ilUyHizRpB5s/view?usp=sharing)
* [image-UDASS(CNN) for Synthia→Cityscapes](https://drive.google.com/file/d/1JgStA157b85Ueh7S_WJUP6iMHhUkLybW/view?usp=sharing)

The checkpoints should be placed in ./work_dirs. Please note that:

* For Synthia→Cityscapes, it is necessary to convert the mIoU to the 16 valid classes. Please, read the
  section above for converting the mIoU.

## Acknowledgements

Image-UDASS is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC/tree/master)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
