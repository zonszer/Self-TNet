# Self-Supervised Sample Difficulty Balancing for Local Descriptor Learning
This repository contains the code implementation for the paper: ["Self-Supervised Sample Difficulty Balancing for Local Descriptor Learning"](https://proceedings.mlr.press/v222/zhang24c/zhang24c.pdf).

## Requirements
Please run the following command to install the required packages and make the necessary directories:
```bash
conda env create -f environment.yml
mkdir HP_descs Models Datasets
```

## Training the Model
To initiate the model training, it's essential to first download the datasets. You should execute the following command to download the Liberty, NotreDame, and Yosemite datasets:

```bash
python -utt PhotoTour.py
```

To train the model on the Liberty, NotreDame, and Yosemite datasets, you can execute the following command. For instance, to train the model on the Liberty dataset, set `--ds=lib`:

```bash
python -utt ftrain.py --id=selfTN_stage1 --arch=SDGMNet128 --ds=lib --loss=ExpTeacher --optimizer=adam --patch_gen=new --sigmas_v=e011 --weight_function=Hessian --epochs=25 --tuples=4050000 --patch_sets=30000 --batch_size=2560 --min_sets_per_img=-1 --max_sets_per_img=1729 --lr=0.0033 --R=1.0 --B=3.7 --Npos=3 --resume='' --teacher=self --A=1.05 --use_stB --threshold=-0.55 --upper=0.10 --all_info 
```
Similarly, replace `--ds=lib` with `--ds=nd` or `--ds=yos` to train the model on the NotreDame and Yosemite datasets, respectively.

Upon completion, execute the following command to fine-tune the model during the annealing training phase:

```bash
python -utt ftrain.py --id=selfTN_stage2 --arch=SDGMNet128 --ds=lib --loss=tripletMargin++ --optimizer=adam --patch_gen=new --sigmas_v=e011 --weight_function=Hessian --epochs=1 --patch_sets=30000  --min_sets_per_img=-1 --max_sets_per_img=1729 --resume=id:selfTN_stage1_XXXXXXX(save_name) --lr=0.0000015 --bsNum=1400 --batch_size=2944 --R=1.0 --B=4.25 --threshold=-0.10 --use_finetune --range=0.0000001 --lr_factor=0.75
```
In the command above, `--resume=id:selfTN_stage1_XXXXXXX` is to set the path to the model's saved folder (which will be auto-saved during the preliminary training). You can locate the model's saved folder and checkpoint under the 'Model' folder and manually retrieve its folder name to fill in the `--resume=` parameter to initialize the annealing training process. 


## Evaluation
Before the evaluation, you should first download and set up the HPatches dataset following the instructions in [HPatches](https://github.com/hpatches/hpatches-benchmark/tree/master/python) repository.

After annealing training, you can evaluate the model on the HPatches benchmark with the following command:

```bash
python -utt HPatches_extract_HardNet.py --model_name=id:selfTN_stage2_XXXXXXX(save_name) --hpatches_dir=(where you download the HPatches dataset)
```
This will generate processed descriptors in the 'HP_descs/(model's saved folder)'. Move the processed descriptors (model's saved folder) to the directory where you have stored the HPatches dataset, for example, 'hpatches-benchmark/data/descriptors'. Then, execute the following command in the 'hpatches-benchmark' directory to evaluate the model on the HPatches benchmark:

```bash
python hpatch_results.py --descr-name=id:selfTN_stage2_XXXXXXX(save_name)
--task=verification --task=retrieval --task=verification
--delimiter=","
--split=full
```

## Reference
If you find this code useful, please consider citing our paper:
```
@InProceedings{pmlr-v222-zhang24c,
  title = 	 {Self-supervised Example Difficulty Balancing for Local Descriptor Learning},
  author =   {Zhang, Jiahan and Tian, Dayong and Wu, Tianyang and Cao, Yiqing and Du, Yaoqi and Wei, Yiwen},
  booktitle =  {Proceedings of the 15th Asian Conference on Machine Learning},
  pages = 	 {1654--1669},
  year = 	   {2024},
  editor = 	 {Yanıkoğlu, Berrin and Buntine, Wray},
  volume = 	 {222},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {11--14 Nov},
  publisher =    {PMLR},
}
