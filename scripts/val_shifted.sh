python_path="/home/Users/dqy/miniconda3/envs/SPIC/bin/python"

cd /home/Users/dqy/Projects/SPIC/
qp=35
save_folder="/home/Users/dqy/Projects/SPIC/Result/shifted/series/qp=${qp}/"
CUDA_VISIBLE_DEVICES=7 ${python_path} image_sample.py \
	--data_dir ./data/cityscapes_sequences \
	--dataset_mode cityscapes --batch_size 1 \
	--attention_resolutions 32,16,8 \
	--diffusion_steps 1000 --image_size 256 \
	--learn_sigma True --noise_schedule linear \
	--num_channels 256 --num_head_channels 64 \
	--num_res_blocks 2 --resblock_updown True \
	--use_fp16 True --use_scale_shift_norm True \
	--num_classes 19 --class_cond True \
	--large_size 128 --small_size 64 \
	--no_instance True --num_samples 500 \
	--s 1.5 --max_iter_time 200 \
	--timestep_respacing 100 \
	--model_path ./checkpoints/shifted/model.pt \
	--compression_level ${qp} \
	--results_path "${save_folder}" \
	--series True