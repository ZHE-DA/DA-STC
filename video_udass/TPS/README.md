## Installation

1. create conda environment

```bash
conda create -n TPS python=3.6
conda activate TPS
conda install -c menpo opencv
pip install torch==1.2.0 torchvision==0.4.0
```

2. clone the [ADVENT repo](https://leftgithub.com/valeoai/ADVENT)

```bash
git clone https://github.com/valeoai/ADVENT
pip install -e ./ADVENT
```

3. clone the current repo

```bash
git clone https://github.com/xing0047/TPS.git
pip install -r ./TPS/requirements.txt
```

4. resample2d dependency:

```
python ./TPS/tps/utils/resample2d_package/setup.py build
python ./TPS/tps/utils/resample2d_package/setup.py install
```

## Data Preparation

1. [Cityscapes-Seq](https://www.cityscapes-dataset.com/)

```
TPS/data/Cityscapes/
TPS/data/Cityscapes/leftImg8bit_sequence/
TPS/data/Cityscapes/gtFine/
```

2. [VIPER](https://playing-for-bencVhmarks.org/download/)

```
TPS/data/Viper/
TPS/data/Viper/train/img/
TPS/data/Viper/train/cls/
```

3. [Synthia-Seq](http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-DAWN.rar)

```
TPS/data/SynthiaSeq/
TPS/data/SynthiaSeq/SEQS-04-DAWN/
```

## Checkpoints

Below, we provide checkpoints of UDASS for the different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

* [UDASS (VIT) for Synthia-Seq→Cityscapes-Seq](https://drive.google.com/file/d/1kwzpghUD1UiK6AvQyazSw0gMGYjAUCwe/view?usp=sharing)
* [UDASS (CNN) for Synthia-Seq→Cityscapes-Seq](https://drive.google.com/file/d/1XJ5naWs9wuZ8k1r6VRHx6YRF7gK5HuCV/view?usp=sharing)
* [UDASS (VIT) for Viper →Cityscapes-Seq](https://drive.google.com/file/d/1OCDnHlz2lJplnPcV7iINOhiRLUTeNm6P/view?usp=sharing)
* [UDASS (CNN) for Viper →Cityscapes-Seq](https://drive.google.com/file/d/1TGpysDaBkQ3F-NQj9wTL0JJnclQMQmto/view?usp=sharing)

## Optical Flow Estimation

For quick preparation, please download the estimated optical flow of all datasets here.

- Synthia-Seq

  [train](https://drive.google.com/file/d/18q6KH-beoBp5jSr1Pl1lMiEcb2te2vxq/view?usp=sharing)
- VIPER

  [train](https://drive.google.com/file/d/1aOeyBLECPSW_ujMBE9RXKjVhTbhw4L2O/view?usp=sharing)
- Cityscapes-Seq

  [train](https://drive.google.com/file/d/193uZifde7WiuImwAgshkPTt1Z6zgE3z8/view?usp=sharing) | [val](https://drive.google.com/file/d/1USizndlUewVb8Eqh4SV6uNuLCEfV9vzU/view?usp=sharing)

## Train and Test

- Train

```
  cd tps/scripts
  CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tps_syn2city.yml
  CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tps_viper2city.yml
```

- Test (may in parallel with Train)

```
  cd tps/scripts
  CUDA_VISIBLE_DEVICES=1 python test.py --cfg configs/tps_syn2city.yml
  CUDA_VISIBLE_DEVICES=1 python test.py --cfg configs/tps_viper2city.yml
```

## Acknowledgement

This codebase is heavily borrowed from [TPS](https://github.com/xing0047/tps).
