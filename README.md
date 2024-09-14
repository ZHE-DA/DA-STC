# UDASS
The official repo for "Unified Domain Adaptive Semantic Segmentation". [[paper](https://arxiv.org/abs/2311.13254)]        
(Arxiv V1 title: "DA-STC: Domain Adaptive Video Semantic Segmentation via Spatio-Temporal Consistency")

The model and test code is unloaded, and we promise that the train code will be uploaded soon!


# Abstract
Unsupervised Domain Adaptive Semantic Segmentation (UDA-SS) aims to transfer the supervision from a labeled source domain to an unlabeled target domain. The majority of existing UDA-SS works typically consider images whilst recent attempts have extended further to tackle videos by modeling the temporal dimension. Although the two lines of research share the major challenges -- overcoming the underlying domain distribution shift, their studies are largely independent. This causes several issues: (1) The insights gained from each line of research remain fragmented, leading to a lack of a holistic understanding of the problem and potential solutions. (2) Preventing the unification of methods, techniques, and best practices across the two domains, resulting in redundant efforts and missed opportunities for cross-pollination of ideas. (3) Without a unified approach, the knowledge and advancements made in one domain (images or videos) may not be effectively transferred to the other, leading to suboptimal performance and slower progress.
Under this observation, we advocate unifying the study of UDA-SS across video and image scenarios, enabling a more comprehensive understanding, synergistic advancements, and efficient knowledge sharing.
To that end, we explore the unified UDA-SS from a general data augmentation perspective, serving as a unifying conceptual framework, enabling improved generalization, and potential for cross-pollination of ideas, ultimately contributing to the overall progress and practical impact of this field of research. Specifically, we propose a Quad-directional Mixup (QuadMix) method, characterized by tackling distinct point attributes and feature inconsistencies through four-directional paths for intra- and inter-domain mixing in a feature space. To deal with temporal shifts with videos, we incorporate optical flow-guided feature aggregation across spatial and temporal dimensions for fine-grained domain alignment. Extensive experiments show that our method outperforms the state-of-the-art works by large margins on four challenging UDA-SS benchmarks. Our source code and models will be released at https://github.com/ZHE-SAPI/UDASS. 

*Index Terms: Unified domain adaptation, semantic segmentation, QuadMix, flow-guided spatio-temporal aggregation.*
  

[![Watch the video](http://img.youtube.com/vi/DgrZYkebhs0/0.jpg)](https://youtu.be/DgrZYkebhs0)

# Installation
1. create conda environment  
```
    conda create -n TPS python=3.6  
    conda activate TPS  
    conda install -c menpo opencv  
    pip install torch==1.2.0 torchvision==0.4.0
```

2.clone the ADVENT repo  
``` 
    git clone https://github.com/valeoai/ADVENT  
    pip install -e ./ADVENT
```

3. clone the current repo   
``` 
    git clone https://github.com/ZHE-SAPI/DA-STC    
    pip install -r ./DASTC/requirements.txt
```

4. resample2d dependency  
``` 
    cd /DASTC/dastc/utils/resample2d_package  
    python setup.py build  
    python setup.py install
``` 

# Data Preparation  
Please refer to the structure of the folder .\video_seg\DASTC\data  
1. [Cityscapes-Seq](https://www.cityscapes-dataset.com/)  
2. [Synthia-Seq](https://synthia-dataset.net/)    
3. [Viper](https://www.playing-for-benchmarks.org/)  

# Pretrained Models  
Download here and put them under  .\DASTC\pretrained_models.  
[SYNTHIA-Seq → Cityscapes-Seq](https://drive.google.com/file/d/1ltMy4ekKczo6saDavQtaZraDwJtWCX9F/view?usp=drive_link)   
|road |side. |buil. |pole |light |sign |vege. |sky| pers. |rider| car| mIOU
| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:|
 |94.1 |61.9| 82.9| 36.9| 41.0 |59.1 |85.2| 85.6 |64.3 |37.8 |90.3 |67.2|
 
[VIPER → Cityscapes-Seq](https://drive.google.com/file/d/1ltMy4ekKczo6saDavQtaZraDwJtWCX9F/view?usp=drive_link)     
|road |side. |buil. |fenc. |light |sign |vege. |terr. |sky |pers. |car| truc.| bus| mot.| bike| mIOU
| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:|
|87.3| 43.8|87.3 |25.2 |40.0| 36.9| 86.7| 20.8 |90.3| 65.8| 86.8 |48.6 |65.6| 37.6| 47.9 |58.2|

# Optical Flow Estimation  
Please first refer to [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch), [Nvidia Semantic Segmentation](https://github.com/NVIDIA/semantic-segmentation), the full optical data will be unloaded soon.  


# Train and Test  
1. Train  
```   
cd /DASTC  
python ./dastc/scripts/train_DAVSS_DSF_cd_ablation_syn.py --cfg ./dastc/scripts/configs/dastc_syn2city.yml

python ./dastc/scripts/train_DAVSS_DSF_cd_ablation_viper.py --cfg ./dastc/scripts/configs/dastc_viper2city.yml    
``` 

2. Test
``` 
cd /DASTC  
python ./dastc/scripts/test_DAVSS_DSF_cd_ablation_syn.py --cfg ./dastc/scripts/configs/dastc_syn2city.yml

python ./dastc/scripts/test_DAVSS_DSF_cd_ablation_viper.py --cfg ./dastc/scripts/configs/dastc_viper2city.yml
``` 
 
# Acknowledgement  
This codebase is borrowed from [TPS](https://github.com/xing0047/tps), [DSP](https://github.com/GaoLii/DSP), [ProDA](https://github.com/microsoft/ProDA/tree/main).  


# Citation
```
@misc{zhang2023dastc,
      title={STCL: Spatio-Temporal Consistency Learning for Domain Adaptive Video Semantic Segmentation}, 
      author={Zhe Zhang and Gaochang Wu and Jing Zhang and Chunhua Shen and Dacheng Tao and Tianyou Chai},
      year={2023},
      eprint={2311.13254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
