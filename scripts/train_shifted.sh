python_path="/home/Users/dqy/miniconda3/envs/SPIC/bin/python"
script_path="/home/Users/dqy/Projects/SPIC/image_train.py"
dataset_path="/home/Users/dqy/Projects/SPIC/data/cityscapes_sequences/"
save_path="/home/Users/dqy/Projects/SPIC/checkpoints/shifted/"
gpu="4,5"

cd /home/Users/dqy/Projects/SPIC/

CUDA_VISIBLE_DEVICES=${gpu} $python_path $script_path --data_dir ${dataset_path} \
	--dataset_mode cityscapes --lr 1e-4 --batch_size 8 \
	--attention_resolutions 32,16,8 --diffusion_steps 1000  \
	--image_size 256 --learn_sigma True --noise_schedule linear \
	--num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
	--resblock_updown True --use_fp16 True --use_scale_shift_norm True \
	--use_checkpoint True --num_classes 19 --class_cond True \
	--large_size 128 --small_size 64 --no_instance True \
	--resume_checkpoint ./checkpoints/pretrained/model.pt \
	--save_dir ${save_path} \
	--shifted True