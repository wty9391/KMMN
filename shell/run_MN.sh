#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=(0)
advertisers=(1458 2259 2261 2821 2997 3358 3386 3427 3476)

for i in ${!advertisers[@]}; do
    advertiser=${advertisers[$i]}
    cuda=${CUDA_VISIBLE_DEVICES[$(($i % ${#CUDA_VISIBLE_DEVICES[@]}))]}
    mkdir -p ../result/$advertiser/log/MN
    mkdir -p ../result/$advertiser/evaluate
    echo "run [python ../run_MN.py ../result/$advertiser $cuda]"
    python ../run_MN.py ../result/$advertiser $cuda\
        1>"../result/$advertiser/log/MN/1.log" 2>"../result/$advertiser/log/MN/2.log"&
done

