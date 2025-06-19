# Enformer_Borzoi_Training_PyTorch

Enformer ([https://www.nature.com/articles/s41592-021-01252-x](https://www.nature.com/articles/s41592-021-01252-x)) and Borzoi ([https://www.nature.com/articles/s41588-024-02053-6](https://www.nature.com/articles/s41588-024-02053-6)) are highly influential genomics models. Unfortunately, they are both implemented in TensorFlow, making them less accessible for most researchers. Fortunately, recent PyTorch implementations of both models have been made available ([enformer-pytorch](https://github.com/lucidrains/enformer-pytorch), [borzoi-pytorch](https://github.com/johahi/borzoi-pytorch/tree/main)). However, these repositories do not provide training scripts. To facilitate community adoption, we provide H5-formatted Enformer datasets and corresponding training code for both Enformer and Borzoi models based on the Enformer data.

## Core Features

We provide the following key features:

1. **Training code based on Hugging Face Trainer** - Ready-to-use training scripts with minimal modifications
2. **Enformer dataset in H5 format** - Converted from original TensorFlow format for easier access

To demonstrate minimal modifications to the original HF Trainer, the provided code trains exclusively on human data.

## Dataset

We utilize the dataset from [Basenji](https://console.cloud.google.com/storage/browser/basenji_barnyard), which is originally in TensorFlow data format and requires users to pay for download costs. We have converted the data to H5 format and made it freely available for download on 🤗 **Hugging Face**: [https://huggingface.co/datasets/yangyz1230/space](https://huggingface.co/datasets/yangyz1230/space).

Please note that the H5 format is compressed but not suitable for parallel DataLoader access. We recommend converting the H5 data to NumPy `.npy` format for better performance. We have implemented both H5 and NPY dataset classes in the `dataloaders/` directory. While our training scripts use the NPY format dataset by default, you can easily switch to the H5 dataset implementation if needed.

## Training Scripts

**Enformer training:**
```bash
bash train_enformer.sh
```
**Borzoi training:**
```
bash train_borzoi.sh
```

## Acknowledgements

Our implementation is based on [enformer-pytorch](https://github.com/lucidrains/enformer-pytorch) and [borzoi-pytorch](https://github.com/johahi/borzoi-pytorch). We thank the authors for their excellent work.

## Citation

This code is extracted and organized from our recent project **SPACE** ([arXiv:2506.01833](https://arxiv.org/abs/2506.01833)). For the complete implementation, please visit our main repository: [https://github.com/ZhuJiwei111/space](https://github.com/ZhuJiwei111/space).

If you find the code in this repository useful, we would be very grateful if you could consider citing our paper:

```bibtex
@misc{yang2025spacegenomicprofilepredictor,
     title={SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model}, 
     author={Zhao Yang and Jiwei Zhu and Bing Su},
     year={2025},
     eprint={2506.01833},
     archivePrefix={arXiv},
     primaryClass={cs.LG},
     url={https://arxiv.org/abs/2506.01833}, 
}
```
