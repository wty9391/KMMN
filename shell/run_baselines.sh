#!/usr/bin/env bash
ipinyou="../datasets/make-ipinyou-data/"
advertisers="1458 2259 2261 2821 2997 3358 3386 3427 3476"

for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/run_baselines
    echo "run [python ../run_baselines.py ../result/$advertiser $ipinyou]"
    python ../run_baselines.py ../result/$advertiser $ipinyou\
        1>"../result/$advertiser/log/run_baselines/1.log" 2>"../result/$advertiser/log/run_baselines/2.log"&
done

#criteo="../datasets/make-criteo-data/"
#advertiser="criteo"
#mkdir -p ../result/$advertiser/log/run_baselines
#echo "run [python ../run_baselines.py ../result/$advertiser $criteo]"
#python ../run_baselines.py ../result/$advertiser $criteo\
#    1>"../result/$advertiser/log/run_baselines/1.log" 2>"../result/$advertiser/log/run_baselines/2.log"&