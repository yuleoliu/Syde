testsets=('bird200' 'pet37' 'car196' 'food101')
gpu=3
OODs=('iNaturalist' 'SUN' 'Texture' 'Places')

methods=('zs_noisytta_configs.py')

for testset in "${testsets[@]}"; do
    for OOD in "${OODs[@]}"; do
        for method in "${methods[@]}"; do
            echo "Running experiment: ID=${testset}, OOD=${OOD}"
            python ./main.py \
                --config configs/$method \
                --test_set $testset \
                --OOD_set $OOD \
                --gpu $gpu 
        done
    done
done