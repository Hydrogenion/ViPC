# ViPC: View-Guided Point Cloud Completion
PyTorch implementation of ViPC (View-Guided Point Cloud Completion)

## Introduction
This work is accepted by CVPR 2021. We present a view-guided solution for the task of point cloud completion. Unlike most existing methods directly inferring the missing points using shape priors, we address this task by introducing ViPC (view-guided point cloud completion) that takes the missing crucial global structure information from an extra single-view image. Besides, we build a large-scale dataset for the point cloud completion task on the ShapeNet dataset. This dataset simulates point cloud defects caused by various kinds of occlusions.You can also check out paper for a deeper introduction.

## Citation
if you find our work useful in your research, please consider citing:
```
@inproceedings{zhang2021vipc,
  title={View-Guided Point Cloud Completion},
  author={Zhang, Xuancheng and Feng, Yutong and Li, Siqi and Zou, Changqing and Wan, Hai and Zhao, Xibin and Guo, Yandong and Gao, Yue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15890--15899},
  year={2021}
}
```

## Usage
### Installation
Install PyTorch over 1.4. The code has been tested with Python 3.7, PyTorch 1.7.0 and CUDA 10.1 on Ubuntu 20.04 and GTX 1080Ti.
Note that RTX graphics card with Tensor Core may not be able to run part of the code.

### Data Prepare
First, please download the [ShapeNetViPC-Dataset](https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH28A) (143GB, code: **ar8l**). Then run ``cat ShapeNetViPC-Dataset.tar.gz* | tar zx``, you will get ``ShapeNetViPC-Dataset`` contains three floders: ``ShapeNetViPC-Partial``, ``ShapeNetViPC-GT`` and ``ShapeNetViPC-View``. 

For each object, the dataset include partial point cloud (``ShapeNetViPC-Patial``), complete point cloud (``ShapeNetViPC-GT``) and corresponding images (``ShapeNetViPC-View``) from 24 different views. You can find the detail of 24 cameras view in ``/ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt``.

<!-- Tip: A [small dataset]() can help you to verify your code. -->

``unzip train_test_list.zip``, you will get train and test list.
Use the code in  ``utils/dataloader.py`` to load the dataset.

The remaining code will be open source soon!!!

## License
Our code is released under MIT License (see LICENSE file for details).
