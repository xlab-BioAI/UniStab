# UniStab: A unified predictor of protein stability changes across all mutation types

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)
[![Hydra](https://img.shields.io/badge/Hydra-1e90ff?logo=dropbox&logoColor=white)](https://github.com/facebookresearch/hydra)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-B31B1B)](https://arxiv.org/) <!-- TODO: 更新论文链接 -->

![UniStab model overview](assets/model.png)

## Introduction

UniStab is a unified deep learning framework that predicts protein stability changes across **single-point**, **multi-point**, and **indel** mutations in an end-to-end manner. 

## Installation
```bash
git clone https://github.com/xtanh/UniStab.git  
```

## Requirements
```bash
conda env create -f environment.yml  
conda activate UniStab
```

## Downloading weights
Download the pre-trained model weights from [Google Drive](https://drive.google.com/file/d/1NEe60FXqpmwLi445SJ93d9_Dpqhdy7YA/view?usp=sharing) and place them in the appropriate directory.



## Training
To train the model with default configurations:

```bash
python src/train_lightning.py
```
You can modify training parameters in `config/default.yaml`.

## Inference
```bash
sh infe.sh
```

## Data

## License


## Acknowledgments

We gratefully acknowledge the following projects and their contributions:

- **ThermoMPNN-D**: Transfer learning framework for protein stability prediction ([Dieckhaus et al., 2024](https://www.pnas.org/doi/abs/10.1073/pnas.2314853121))
- **ESMFold**: Evolutionary-scale protein structure prediction ([Lin et al., 2023](https://www.science.org/doi/10.1126/science.ade2574))

Parts of the codebase are adapted from these excellent works.

## Citation
```bibtex
% TODO: 论文引用条目
```

## Reference
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}

@article{dieckhaus2025protein,
  title={Protein stability models fail to capture epistatic interactions of double point mutations},
  author={Dieckhaus, Henry and Kuhlman, Brian},
  journal={Protein Science},
  volume={34},
  number={1},
  pages={e70003},
  year={2025},
  publisher={Wiley Online Library}
}
```

