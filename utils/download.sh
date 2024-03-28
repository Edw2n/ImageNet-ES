# sh ./utils/dowanload.sh

# download the up-to-date benchmarks and checkpoints
# provided by OpenOOD v1.5
python ./utils/download.py \
	--contents 'datasets' \
	--datasets 'ood_v1.5' \
	--save_dir './data' \
	--dataset_mode 'benchmark'
