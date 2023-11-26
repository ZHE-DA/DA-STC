# STCL
The official repo for "STCL: Spatio-Temporal Consistency Learning for Domain Adaptive Video Semantic Segmentation". [[paper](https://arxiv.org/abs/2311.13254)]        
The model and test code is unloaded, and the train code will be uploaded soon! 


# Abstract
Video semantic segmentation is a pivotal aspect of video representation learning. However, significant domain shifts present a challenge in effectively learning invariant spatio-temporal features across the labeled source domain and unlabeled target domain for video semantic segmentation. To solve the challenge, we propose a novel STCL method for domain adaptive video semantic segmentation, which incorporates a bidirectional multi-level spatio-temporal fusion module and a category-aware spatio-temporal feature alignment module to facilitate consistent learning for domain-invariant features. Firstly, we perform bidirectional spatio-temporal fusion at the image sequence level and shallow feature level, leading to the construction of two fused intermediate video domains. This prompts the video semantic segmentation model to consistently learn spatio-temporal features of shared patch sequences which are influenced by domain-specific contexts, thereby mitigating the feature gap between the source and target domain. Secondly, we propose a category-aware feature alignment module to promote the consistency of spatio-temporal features, facilitating adaptation to the target domain. Specifically, we adaptively aggregate the domain-specific deep features of each category along spatio-temporal dimensions, which are further constrained to achieve cross-domain intra-class feature alignment and inter-class feature separation. Extensive experiments demonstrate the effectiveness of our method, which achieves state-of-the-art mIOUs on multiple challenging benchmarks. Furthermore, we extend the proposed STCL to the image domain, where it also exhibits superior performance for domain adaptive semantic segmentation. The source code and models will be made available at [STCL](https://github.com/ZHE-SAPI/STCL).

*Index Terms: Domain adaptation, video semantic segmentation, feature alignment, spatio-temporal consistency.*
  

# A short video demo of our results:
https://github.com/ZHE-SAPI/DA-STC/assets/52643313/d5738a66-8e77-4718-a851-4218d500f800

A longer video demo for [more cases](https://drive.google.com/file/d/1lPZkvsY3rBFVRjNlrz4h2xGsD41J7Y5Z/view?usp=drive_link) is available.
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
 |90.8 |39.9| 83.2| 33.2| 30.1 |50.7 |84.8| 82.3 |61.2 |32.7 |87.4 |61.5|
 
[VIPER → Cityscapes-Seq](https://drive.google.com/file/d/1ltMy4ekKczo6saDavQtaZraDwJtWCX9F/view?usp=drive_link)     
|road |side. |buil. |fenc. |light |sign |vege. |terr. |sky |pers. |car| truc.| bus| mot.| bike| mIOU
| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:| :----:|
|91.6| 51.4 |87.0 |24.1 |32.3| 37.2| 84.1| 28.4 |84.8| 64.4| 85.7 |41.4 |46.5| 34.0| 49.6 |56.2|

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
