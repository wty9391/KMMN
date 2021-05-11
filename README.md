# Combining Kaplan-Meier Estimator with Markov Network to Predict Market Price Distribution in Online Advertising
This is a repository of experiment code supporting [Combining Kaplan-Meier Estimator with Markov Network to Predict Market Price Distribution in Online Advertising]().

For any problems, please report the issues here.

## Quirk Start

### Prepare Dataset
Before run the demo, please first check the GitHub project [make iPinYou data](https://github.com/wnzhang/make-ipinyou-data) for pre-processing the [iPinYou dataset](http://data.computational-advertising.org).
Or you can download the processed dataset from this [link](https://pan.baidu.com/s/1bjeROrEuxouy9Mhfd1vrCw) with extracting code `h12c`.

Then, please create a folder named `dataset`, and put the dataset in it.
The file tree looks like this:
```
KMMN
│───README.md
│
└───dataset
│   └───make-ipinyou-data
│       │   1458
│       │   2259
│       │   ...
        make-criteo-data
        |  criteo_attribution_dataset.tsv.gz
...
```

### Requirements

Our experimental platform's key configuration:
* an Intel(R) Xeon(R) E5-2620 v4 CPU processor
* a NVIDIA TITAN Xp GPU processor
* 64 GB memory
* Ubuntu 18.04 operating system
* python 3.6
* pytorch 1.3
* cuda 10.1

Required Python libraries are listed in `./requirements.txt`



### Encode Dataset
Please run the following code to encode the dataset
```bash
cd ./shell
bash ./init_dataset.sh
```
You can find the running logs in this directory `/result/$advertiser/log/dataset_encode`

### Run KMMN
Please check the CUDA_VISIBLE_DEVICES variable in `./run_MN.sh` before running.

Then run the following code to train and evaluate the KMMN
```bash
bash ./run_MN.sh
```
You can find the running logs in this directories `/result/$advertiser/log/run_MN`

### Run Baselines
Please run the following code to train and evaluate the baselines
```bash
bash ./run_baselines.sh
```
The performance of all baselines can be found in `/result/baseline_report.csv`

You can also find the running logs in these directories `/result/$advertiser/log/run_baselines`

The code of DCL and DLF can be forked from this [repository](https://github.com/rk2900/DLF).







