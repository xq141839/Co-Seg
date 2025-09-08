# [MICCAI 2025] Co-Seg: Mutual Prompt-Guided Collaborative Learning for Tissue and Nuclei Segmentation

### [ArXiv Paper]() 

[Qing Xu](https://scholar.google.com/citations?user=IzA-Ij8AAAAJ&hl=en&authuser=1)<sup>1,2</sup> [Wenting Duan](https://scholar.google.com/citations?user=H9C0tX0AAAAJ&hl=en&authuser=1)<sup>2</sup> [Zhen Chen](https://franciszchen.github.io/)<sup>3✉</sup> 

<sup>1</sup>University of Lincoln &emsp; <sup>2</sup>Univeristy of Nottingham &emsp; <sup>3</sup>Yale &emsp;

<sup>✉</sup> Corresponding Author. 

-------------------------------------------
![introduction](method.png)

## 📰News

- **[2025.08.09]** We have released the code for Co-Seg !
## 🛠Setup

```bash
git clone https://github.com/xq141839/Co-Seg.git
cd Co-Seg
```

**Key requirements**: Cuda 12.2+, PyTorch 2.4+

## 📚Data Preparation
- **PUMA**: [Challenge Link](https://puma.grand-challenge.org/)

The data structure is as follows.
```
Co-Seg
├── datasets
│   ├── image_1024
│     ├── training_set_metastatic_roi_001.png
|     ├── ...
|   ├── mask_sem_1024
│     ├── training_set_metastatic_roi_001_nuclei.npy
|     ├── ...
|   ├── mask_ins_1024
│     ├── training_set_metastatic_roi_001_tissue.npy
|     ├── ...
|   ├── data_split.json
```
The json structure is as follows.

    { 
     "train": ['training_set_metastatic_roi_061.png'],
     "valid": ['training_set_metastatic_roi_002.png'],
     "test":  ['training_set_metastatic_roi_009.png'] 
     }

## 🎪Quickstart
* Train the Co-Seg with the default settings:
```python
python train.py --dataset data/$YOUR DATASET NAME$ --sam_pretrain pretrain/sam2_hiera_large.pth
```

## Acknowledgements

* [SAM2](https://github.com/facebookresearch/sam2)
* [Medical-SAM-Adapter](https://github.com/SuperMedIntel/Medical-SAM-Adapter)

