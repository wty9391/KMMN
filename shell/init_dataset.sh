#!/usr/bin/env bash
ipinyou="../datasets/make-ipinyou-data/"
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427"
#advertisers="2821"
for advertiser in $advertisers; do
    echo "run [python ../ipinyou_init.py $ipinyou/$advertiser/train.log.txt $ipinyou/$advertiser/test.log.txt $ipinyou/$advertiser/featindex.txt ../result/$advertiser]"
    mkdir -p ../result/$advertiser/log/dataset_encode
    python ../ipinyou_init.py $ipinyou/$advertiser/train.log.txt $ipinyou/$advertiser/test.log.txt $ipinyou/$advertiser/featindex.txt ../result/$advertiser\
        1>"../result/$advertiser/log/dataset_encode/1.log" 2>"../result/$advertiser/log/dataset_encode/2.log"&
done

criteo="../datasets/make-criteo-data/"
mkdir -p ../result/criteo/log/dataset_encode
echo "run [python ../criteo_init.py $criteo/criteo_attribution_dataset.tsv.gz ../result/criteo]"
python ../criteo_init.py $criteo/criteo_attribution_dataset.tsv.gz ./result/criteo \
        1>"../result/criteo/log/dataset_encode/1.log" 2>"../result/criteo/log/dataset_encode/2.log"&

