### Unified Domain Adaptive Semantic Segmentation

## Main Results

#### SYNTHIA-Seq => Cityscapes-Seq

| Methods                | road           | side.          | buil.          | pole           | light          | sign           | vege.          | sky            | per.           | rider          | car            | mIoU           |
| ---------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Source                 | 56.3           | 26.6           | 75.6           | 25.5           | 5.7            | 15.6           | 71.0           | 58.5           | 41.7           | 17.1           | 27.9           | 38.3           |
| DA-VSN                 | 89.4           | 31.0           | 77.4           | 26.1           | 9.1            | 20.4           | 75.4           | 74.6           | 42.9           | 16.1           | 82.4           | 49.5           |
| PixMatch               | 90.2           | 49.9           | 75.1           | 23.1           | 17.4           | 34.2           | 67.1           | 49.9           | 55.8           | 14.0           | 84.3           | 51.0           |
| TPS                    | 91.2           | 53.7           | 74.9           | 24.6           | 17.9           | 39.3           | 68.1           | 59.7           | 57.2           | 20.3           | 84.5           | 53.8           |
| **QuadMix(CNN)** | 90.8           | 39.9           | **83.2** | 33.2           | 30.1           | 50.7           | 84.8           | 83.2           | 61.2           | 32.7           | 87.4           | 61.5           |
| **QuadMix(ViT)** | **94.1** | **61.9** | 82.9           | **36.9** | **41.0** | **59.1** | **85.2** | **85.6** | **64.3** | **37.8** | **90.3** | **67.2** |

#### VIPER => Cityscapes-Seq

| Methods                | road           | side.          | buil.          | fence          | light          | sign           | vege.          | terr.          | sky            | per.           | car            | truck          | bus            | motor          | bike           | mIoU           |
| ---------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Source                 | 56.7           | 18.7           | 78.7           | 6.0            | 22.0           | 15.6           | 81.6           | 18.3           | 80.4           | 59.9           | 66.3           | 4.5            | 16.8           | 20.4           | 10.3           | 37.1           |
| PixMatch               | 79.4           | 26.1           | 84.6           | 16.6           | 28.7           | 23.0           | 85.0           | 30.1           | 83.7           | 58.6           | 75.8           | 34.2           | 45.7           | 16.6           | 12.4           | 46.7           |
| DA-VSN                 | 86.8           | 36.7           | 83.5           | 22.9           | 30.2           | 27.7           | 83.6           | 26.7           | 80.3           | 60.0           | 79.1           | 20.3           | 47.2           | 21.2           | 11.4           | 47.8           |
| TPS                    | 82.4           | 36.9           | 79.5           | 9.0            | 26.3           | 29.4           | 78.5           | 28.2           | 81.8           | 61.2           | 80.2           | 39.8           | 40.3           | 28.5           | 31.7           | 48.9           |
| **QuadMix(CNN)** | **91.6** | **51.4** | 87.0           | 24.1           | 32.3           | **37.2** | 84.1           | **28.4** | 84.8           | 64.4           | 85.7           | 41.4           | 46.5           | 34.0           | 49.6           | 56.2           |
| **QuadMix(ViT)** | 87.3           | 43.8           | **87.3** | **25.2** | **40.0** | 36.9           | **86.7** | 20.8           | **90.3** | **65.8** | **86.8** | **48.6** | **65.6** | **37.6** | **49.7** | **58.2** |

## Environment Setup

1. create conda environment

```bash
conda create -n UDAVSS python=3.6
conda activate UDAVSS
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
pip install -r ./VIDEO/requirements.txt
```

4. resample2d dependency:

```
python ./VIDEO/tps/utils/resample2d_package/setup.py build
python ./VIDEO/tps/utils/resample2d_package/setup.py install
```

## Data Preparation

1. [Cityscapes-Seq](https://www.cityscapes-dataset.com/)

```
VIDEO/data/Cityscapes/
VIDEO/data/Cityscapes/leftImg8bit_sequence/
VIDEO/data/Cityscapes/gtFine/
```

2. [VIPER](https://playing-for-bencVhmarks.org/download/)

```
VIDEO/data/Viper/
VIDEO/data/Viper/train/img/
VIDEO/data/Viper/train/cls/
```

3. [Synthia-Seq](http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-DAWN.rar)

```
VIDEO/data/SynthiaSeq/
VIDEO/data/SynthiaSeq/SEQS-04-DAWN/
```

## Checkpoints

Below, we provide checkpoints of UDASS(HRDA) for the different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

* [video-UDASS(VIT) for Synthia-Seq→Cityscapes-Seq](https://drive.google.com/file/d/1kwzpghUD1UiK6AvQyazSw0gMGYjAUCwe/view?usp=sharing)
* [video-UDASS(VIT) for VIPER→Cityscapes-Seq](https://drive.google.com/file/d/1OCDnHlz2lJplnPcV7iINOhiRLUTeNm6P/view?usp=sharing)
* [video-UDASS(CNN) for Synthia-Seq→Cityscapes-Seq](https://drive.google.com/file/d/1XJ5naWs9wuZ8k1r6VRHx6YRF7gK5HuCV/view?usp=sharing)
* [video-UDASS(CNN) for VIPER→Cityscapes-Seq](https://drive.google.com/file/d/1TGpysDaBkQ3F-NQj9wTL0JJnclQMQmto/view?usp=sharing)

## Optical Flow Estimation

For quick preparation, please download the estimated optical flow of all datasets here, and put them in the ./VIDEO/data folder.

- Synthia-Seq

  [train](https://drive.google.com/file/d/18q6KH-beoBp5jSr1Pl1lMiEcb2te2vxq/view?usp=sharing)
- VIPER

  [train](https://drive.google.com/file/d/1aOeyBLECPSW_ujMBE9RXKjVhTbhw4L2O/view?usp=sharing)
- Cityscapes-Seq

  [train](https://drive.google.com/file/d/193uZifde7WiuImwAgshkPTt1Z6zgE3z8/view?usp=sharing) | [val](https://drive.google.com/file/d/1USizndlUewVb8Eqh4SV6uNuLCEfV9vzU/view?usp=sharing)

  The steps to generate the optical flow (refer to [issue](https://github.com/Dayan-Guan/DA-VSN/issues/1)):

1. git clone git clone -b sdcnet [https://github.com/NVIDIA/semantic-segmentation.git](https://github.com/NVIDIA/semantic-segmentation.git);
2. Unzip the files of [video_udass/Code_for_Optical_Flow_Estimation.zip](https://github.com/ZHE-SAPI/UDASS/blob/main/video_udass/Code_for_Optical_Flow_Estimation.zip) and put them in the folder of sdcnet;
3. Run the shell scripts to generate optical flow:
   ```
   [1] CItyscapes validation set: "python Cityscapes_val_optical_flow_scale512.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir ../../data/Cityscapes --target_dir Cityscapes_val_optical_flow_scale512 --vis --resize 0.5"

   [2] SynthiaSeq train set: "python Estimated_optical_flow_SynthiaSeq_train.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir ../../data/SynthiaSeq/SEQS-04-DAWN/rgb --target_dir Estimated_optical_flow_SynthiaSeq_train --vis --resize 0.533333"

   [3] Viper train set: "python Estimated_optical_flow_Viper_train.py --pretrained ../pretrained_models/sdc_cityscapes_vrec.pth.tar --flownet2_checkpoint ../pretrained_models/FlowNet2_checkpoint.pth.tar --source_dir ../../data/viper --target_dir /home/dayan/gdy/adv/snapshots/Estimated_optical_flow_Viper_train--vis --resize 0.533333"
   ```

## Train and Test

- Train

```
  cd xxxxPATHxxxx/video_udass/VIDEO (Please adjust according to the actual address of your device manually )
  # syn2city CNN
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_24.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # viper2city CNN
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_31.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
  # syn2city ViT
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_24_former.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # tps_viper2city ViT
  python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_31_former.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
```

- Test (can in parallel with Train)

```
  cd xxxxPATHxxxx/video_udass/VIDEO
  # syn2city CNN
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_24.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # viper2city CNN
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_31.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
  # syn2city ViT
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_24_former.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml
  # tps_viper2city ViT
  python ./tps/scripts_ablation/test_DAVSS_DSF_cd_ablation_31_former.py --cfg ./tps/scripts_ablation/configs/tps_viper2city.yml
```

## Acknowledgement

This codebase is is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [TPS](https://github.com/xing0047/TPS/tree/main)
* [DA-VSN](https://github.com/Dayan-Guan/DA-VSN)
* [Sdcnet ](https://github.com/NVIDIA/semantic-segmentation)
* [FlowNet 2](https://github.com/NVIDIA/flownet2-pytorch)
