# UDASS

The official repo for "Unified Domain Adaptive Semantic Segmentation". [[paper](https://arxiv.org/abs/2311.13254)] [[demo](https://drive.google.com/file/d/1OT5GtsbC0CcW6aydBL27ADjve95YE5oj/view?usp=sharing)]

# Abstract

**Unsupervised Domain Adaptive Semantic Segmentation (UDA-SS)** aims to transfer the supervision from a labeled source domain to an unlabeled target domain. The majority of existing UDA-SS works typically consider images whilst recent attempts have extended further to tackle videos by modeling the temporal dimension. Although the two lines of research share the major challenges -- overcoming the underlying domain distribution shift, their studies are largely independent.   



**This causes several issues:** (1) The insights gained from each line of research remain fragmented, leading to a lack of a holistic understanding of the problem and potential solutions. (2) Preventing the unification of methods, techniques, and best practices across the two domains, resulting in redundant efforts and missed opportunities for cross-pollination of ideas. (3) Without a unified approach, the knowledge and advancements made in one domain (images or videos) may not be effectively transferred to the other, leading to suboptimal performance and slower progress.



**Under this observation**, we advocate unifying the study of UDA-SS across video and image scenarios, enabling a more comprehensive understanding, synergistic advancements, and efficient knowledge sharing.  To that end, we explore the unified UDA-SS from a general data augmentation perspective, serving as a unifying conceptual framework, enabling improved generalization, and potential for cross-pollination of ideas, ultimately contributing to the overall progress and practical impact of this field of research. Specifically, we propose a Quad-directional Mixup (QuadMix) method, characterized by tackling distinct point attributes and feature inconsistencies through four-directional paths for intra- and inter-domain mixing in a feature space. To deal with temporal shifts with videos, we incorporate optical flow-guided feature aggregation across spatial and temporal dimensions for fine-grained domain alignment. 



**Extensive experiments** show that our method outperforms the state-of-the-art works by large margins on four challenging UDA-SS benchmarks.


*Index Terms: Unified domain adaptation, semantic segmentation, QuadMix, flow-guided spatio-temporal aggregation.*

# Click for more qualitative results

[![Please watch the video for more qualitative results.](https://github.com/ZHE-SAPI/UDASS/blob/main/Unified-UDASS.jpg?raw=true)](https://youtu.be/DgrZYkebhs0)

The video demo is also avaliable at [bilibili](https://www.bilibili.com/video/BV1ZgtMejErB/?vd_source=ae767173839d1c3a41173ad40cc34d53) or [google drive](https://drive.google.com/file/d/1OT5GtsbC0CcW6aydBL27ADjve95YE5oj/view?usp=sharing). Please select HD quality (1080p) for clearer display.

## UDASS for Image Scenarios

You can find the source code to run image-UDASS on domain-adaptive **IMAGE** semantic segmentation in the subfolder [/image_udass](https://github.com/ZHE-SAPI/UDASS/tree/main/image_udass). For instructions how to set up the environment/datasets and how to train UDASS for image semantic segmentation UDA, please refer to [seg/README.md](https://github.com/ZHE-SAPI/UDASS/blob/main/image_udass/seg/README.md).

## UDASS for Video Scenarios

You can find the source code to run image-UDASS on domain-adaptive **VIDEO **semantic segmentation in the subfolder [/video_udass](https://github.com/ZHE-SAPI/UDASS/tree/main/video_udass). For instructions how to set up the environment/datasets and how to train UDASS for image semantic segmentation UDA, please refer to [VIDEO/README.md](https://github.com/ZHE-SAPI/UDASS/blob/main/video_udass/VIDEO/README.md).

## Citation

If you find MIC useful in your research, please consider citing:

```
@misc{zhang2023dastc,
      title={Unified Domain Adaptive Semantic Segmentation}, 
      author={Zhe Zhang and Gaochang Wu and Jing Zhang and Chunhua Shen and Dacheng Tao and Tianyou Chai},
      year={2023},
      eprint={2311.13254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
