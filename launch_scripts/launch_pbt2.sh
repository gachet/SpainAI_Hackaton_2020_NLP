PID=35674
while ps -p $PID; do sleep 10; done ; nohup python -u launch_run.py --data_dir data_2401 --max_source_len 364 --max_target_len 60 \
                                    --output_dir pbt_0302 --batch_size 4 --eval_batch_size 8 --sortish_sampler True --population_number 4  > log_pbt_0302.txt &