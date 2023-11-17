# Hierarchical cumulative network for unsupervised medical image registration

[NEWS!!]**The [pytorch](https://github.com/YutingHe-list/PC-Reg-RT/tree/main/pytorch) version is opened now!**

We propose the we propose a novel hierarchical cumulative network (HCN), which explicitly considers the optimal similarity position with an effective Bidirectional Asymmetric Registration Module (BARM). The BARM simultaneously learns two asymmetric displacement vector fields (DVFs) to optimally warp both moving images and fixed images to their optimal similar shape along the geodesic path. Furthermore, we incorporate the BARM into a Laplacian pyramid network with hierarchical recursion, in which the moving image at the lowest level of the pyramid is warped successively for aligning to the fixed image at the lowest level of the pyramid to capture multiple DVFs. We then accumulate these DVFs and up-sample them to warp the moving images at higher levels of the pyramid to align to the fixed image of the top level. The entire system is end-to-end and jointly trained in an unsupervised manner. Extensive experiments were conducted on two public 3D Brain MRI datasets to demonstrate that our HCN outperforms both the traditional and state-of-the-art registration methods. To further evaluate the performance of our HCN, we tested it on the validation set of the MICCAI Learn2Reg 2021 challenge. Additionally, a cross-dataset evaluation was conducted to assess the generalization of our HCN. Experimental results showed that our HCN is an effective deformable registration method and achieves excellent generalization performance.

<p align="center"><img width="100%" src="fig/detil.png" /></p>

## Paper
This repository provides the official tensorflow implementation of PC-Reg-RT in the following papers:

**Few-shot Learning for Deformable Medical Image Registration with Perception-Correspondence Decoupling and Reverse Teaching** <br/> 
[Yuting He](http://19951124.academic.site/?lang=en), TianTian Li, Rongjun Ge, Jian Yang, [Youyong Kong](https://cse.seu.edu.cn/2019/0105/c23024a257502/page.htm), Jian Zhu, Huazhong Shu, [Guanyu Yang*](https://cse.seu.edu.cn/2019/0103/c23024a257233/page.htm), [Shuo Li*](http://www.digitalimaginggroup.ca/members/shuo.php) <br/>
Southeast University <br/>
IEEE Journal of Biomedical And Health Informatics ([J-BHI](https://www.embs.org/jbhi/)) <br/>
[Paper](https://ieeexplore.ieee.org/document/9477084) | [Code](https://github.com/YutingHe-list/PC-Reg-RT)

## Available implementation
- [tensorflow/](https://github.com/YutingHe-list/PC-Reg-RT/tree/main/tensorflow)
- [pytorch/](https://github.com/YutingHe-list/PC-Reg-RT/tree/main/pytorch)

## Citation
If you use PC-Reg-RT for your research, please cite our papers:
```
@ARTICLE{9477084,
  author={He, Yuting and Li, Tiantian and Ge, Rongjun and Yang, Jian and Kong, Youyong and Zhu, Jian and Shu, Huazhong and Yang, Guanyu and Li, Shuo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Few-Shot Learning for Deformable Medical Image Registration With Perception-Correspondence Decoupling and Reverse Teaching}, 
  year={2022},
  volume={26},
  number={3},
  pages={1177-1187},
  doi={10.1109/JBHI.2021.3095409}}
```

## Acknowledgments

This research was supported by the National Key Research and Development Program of China (2017YFC0109202), National Natural Science Foundation under grants (31800825, 31571001, 61828101), Excellence Project Funds of Southeast University and Scientific Research Foundation of Graduate School of Southeast University (YBPY2139). We thank the Big Data Computing Center of Southeast University for providing the facility support on the numerical calculations in this paper.
