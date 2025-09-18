testsets=('I')
gpu=2
OODs=('iNaturalist' 'SUN' 'Texture' 'Places')
plpd_thresholds=(0.0007)
deyo_margins=(4.0)
filter_ent=1
filter_plpd=1
gap=1

methods=('zs_noisytta_configs.py')

for testset in "${testsets[@]}"; do
    for OOD in "${OODs[@]}"; do
        for method in "${methods[@]}"; do
            for plpd_threshold in "${plpd_thresholds[@]}"; do
                for deyo_margin in "${deyo_margins[@]}"; do
                    echo "Running experiment: ID=${testset}, OOD=${OOD}"
                    python ./main.py \
                        --config configs/$method \
                        --test_set $testset \
                        --OOD_set $OOD \
                        --gpu $gpu \
                        --plpd_threshold $plpd_threshold \
                        --deyo_margin $deyo_margin \
                        --filter_ent $filter_ent \
                        --filter_plpd $filter_plpd \
                        --gap $gap 
                done
            done
        done
    done
done
