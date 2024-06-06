name=$1
step=$2
test_folder="/home/Users/dqy/Projects/SPIC/Result/${name}/step${step}/samples/"
gt_folder="/home/Users/dqy/Projects/SPIC/Result/${name}/step${step}/images/"
save_path="/home/Users/dqy/Projects/SPIC/Result/${name}/step${step}/lpips.json"
/home/Users/dqy/miniconda3/envs/SKB/bin/python \
  "/home/Users/dqy/Projects/SKB-compression/scripts/analysis_LPIPS.py" \
  --folder "$test_folder" \
  --gt_folder "${gt_folder}" \
  --save_path "${save_path}" \
  --size_w 512 \
  --size_h 256 \
  --print
