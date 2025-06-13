rm -rf datasets
python3 prep_data.py --dataset ohjoonhee/UsedCarsImageNetGamma --output_dir datasets --split fold1of5
python3 -u src/main.py fit -c configs/1.yaml $@

rm -rf datasets
python3 prep_data.py --dataset ohjoonhee/UsedCarsImageNetGamma --output_dir datasets --split fold2of5
python3 -u src/main.py fit -c configs/2.yaml $@

rm -rf datasets
python3 prep_data.py --dataset ohjoonhee/UsedCarsImageNetGamma --output_dir datasets --split fold3of5
python3 -u src/main.py fit -c configs/3.yaml $@

rm -rf datasets
python3 prep_data.py --dataset ohjoonhee/UsedCarsImageNetGamma --output_dir datasets --split fold4of5
python3 -u src/main.py fit -c configs/4.yaml $@

rm -rf datasets
python3 prep_data.py --dataset ohjoonhee/UsedCarsImageNetGamma --output_dir datasets --split fold5of5
python3 -u src/main.py fit -c configs/5.yaml $@