# XDFusion
This repository contains the code for the following ACL 2023 main conference paper:

> **Towards Zero-Shot Multilingual Transfer for Code-Switched Responses**. <br>
> Ting-Wei Wu, Changsheng Zhao, Ernie Chang, Yangyang Shi, Pierce I-Jen Chuang, Vikas Chandra and Biing-Hwang Juang <br>
> The 61st Annual Meeting of the Association for Computational Linguistics (**ACL**) (**Outstanding Paper Award**) (Oral Presentation), Jul 2023.

PDF: https://aclanthology.org/2023.acl-long.417.pdf <br>
Resources: https://virtual2023.aclweb.org/paper_P693.html

Currently the framework supports to run on [GlobalWOZ dataset](https://github.com/bosheng2020/globalwoz), which has two types of testing data F&F, F&E.


## Quick Setup

### Environment Setup
1. Install miniconda or anaconda.
2. Create a conda environment and activate it:
    >
        conda create -n test python=3.8
        conda activate test
3. Install packages
    >
        bash install.sh
4. Download the dataset:
    1) Processed data: https://drive.google.com/file/d/18-bHd6lfix1BevSLycGamfI2fKyilgMU/view?usp=drive_link
    2) Original source: https://github.com/bosheng2020/globalwoz

### Train & test XDFusion
To reproduce the experiments, use the average of 3 seed runs: 557, 100, 200
1. Pretrain language adapters: (Specify language and number of data in `pretrain/run_clm_adapter.py`). Then run:
    >
        bash run_clm.sh
2. First-round/Second-round training: 
   Run the commands specified in `run_fs.sh`
3. [Optional] Other testing examples: `run.sh`


## Instructions

>
    python train_t5.py --task             [whether to use DST task: dst]
                       --mode             [type: train/test/score/repr]
                       --datause          [main language to train/test: en/zh/es/id]
                       --fewshot_data     [few-shot language to train: zh/es/id]
                       --fewshot          [number of few-shot data use]
                       --model_checkpoint [pretrained base model path]
                       --model_name       [which base model to use]
                       --sub_model        [which folder to save the model]
                       --postfix          [postfix of checkpoint to save]
                       --GPU              [GPU to use]
                       --n_epochs         [Number of epochs]
                       --t5_span_pretrain [Whether to use MC4 data]
                       --nmt_pretrain     [Whether to use CCMatrix data]
                       --translate_train  [Whether to use translated data]
                       --pretrain_seq2seq [The first stage of training adapter-based model]
                       --fe               [Whether to use F&E dataset]
                       --seed             [Random seed]

## Citation

>
    @inproceedings{wu-etal-2023-towards,
        title = "Towards Zero-Shot Multilingual Transfer for Code-Switched Responses",
        author = "Wu, Ting-Wei  and
        Zhao, Changsheng  and
        Chang, Ernie  and
        Shi, Yangyang  and
        Chuang, Pierce  and
        Chandra, Vikas  and
        Juang, Biing",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-long.417",
        pages = "7551--7563"
    }
