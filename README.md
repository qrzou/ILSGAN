## ILSGAN: Independent Layer Synthesis for  Unsupervised Foreground-Background Segmentation
This is the official implementation of ILSGAN paper [[arXiv]](https://arxiv.org/abs/2211.13974) [AAAI 2023] [Oral].

### Environment
We follow the environment of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch): PyTorch 1.7.1, Python 3.7, CUDA 11.0.
You may also need to install these python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`

### Dataset
We use the `CUB`, `Dog`, and `Car` datasets provided by [DRC](https://github.com/yuPeiyu98/Deep-Region-Competition).
To get these datasets, you can refer to DRC for details. 
* Different from DRC, we use the original image (not resized image) for training ILSGAN. 
So, for the Dogs dataset, you need to also download the `dogs_raw_data.zip` from DRC, 
extract the zip and put the raw Dog data into the Dogs directory.

The directory of dataset should be like as:
```
DRC_processed
├── birds
│   ├── bounding_boxes.txt
│   ├── images
│   ├── images.txt
│   ├── segmentations
│   └── train_val_test_split.txt
├── cars
│   ├── data_mrcnn
│   ├── data_mrcnn_test_select.npy
│   └── data_mrcnn_train_select.npy
└── dogs
    ├── data_mrcnn
    ├── data_te_sel.npy
    ├── data_tr_sel.npy
    ├── test_orig
    └── train_orig
```
Remember to change the path config in the train_ILS_64/128.py to your own directory.

### Getting started

For the experiments under 64*64 resolution, you can simply run the following command.
It can automatically do ILSGAN's training, data generation, segmentation eval, and MI eval.
Select --data option in [car, cub, dog] for your needed dataset.
```.bash
CUDA_VISIBLE_DEVICES=0 python train_ILS_64.py --outdir=./outputs --data=car --gpus=1 --cfg=ILS_predL --batch=32
```

For the experiments under 128*128, you need to manually run the following commands for training, generation, and evaluation:
```.bash
# Train ILSGAN
CUDA_VISIBLE_DEVICES=0 python train_ILS_128.py --outdir=./outputs --data=car --gpus=1 --cfg=ILS --batch=32

# Generate segmentation samples from ILSGAN
CUDA_VISIBLE_DEVICES=0 python generate_segmentation_samples.py --network=./outputs/The-Exp-For-Eval --n=50000 --topk=8000

# Evaluate the segmentation
CUDA_VISIBLE_DEVICES=0 python eval_segmentation_eval128.py --aug=color --syn-data=./outputs/The-Exp-For-Eval/synthetic_data-XXXXXX --real-data=car --scale=128

# Evaluate the mutual information
CUDA_VISIBLE_DEVICES=0 python eval_MI_MINE.py --path=./outputs/The-Exp-For-Eval/auto_test/synthetic_data-XXXXXX

```

If you want to manually eval the 64*64 resolution results, follow these commands:
```.bash
# Generate segmentation samples from ILSGAN
CUDA_VISIBLE_DEVICES=0 python generate_segmentation_samples.py --network=./outputs/The-Exp-For-Eval --n=50000 --topk=8000

# Evaluate the segmentation
CUDA_VISIBLE_DEVICES=0 python eval_segmentation.py --aug=color --syn-data=./outputs/The-Exp-For-Eval/synthetic_data-XXXXXX --real-data=car --scale=64

# Evaluate the mutual information
CUDA_VISIBLE_DEVICES=0 python eval_MI_MINE.py --path=./outputs/The-Exp-For-Eval/auto_test/synthetic_data-XXXXXX
```

### Citation
```
@inproceedings{Zou2023ilsgan,
  title={ILSGAN: Independent Layer Synthesis for Unsupervised Foreground-Background Segmentation},
  author={Zou, Qiran and Yang, Yu and Cheung, Wing Yin and Liu, Chang and Ji, Xiangyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2023}
}
```


