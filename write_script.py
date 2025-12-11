#python -i test_240.py --target_df ../RNA_replicase_target.csv --gpu_id 0 --output_csv replicase_results.csv
import os

os.system("mkdir -p scripts_generate_replicase")

n_gpus = 8
n_runs = 24*60

scripts={i:'' for i in range(n_gpus)}

for i in range(n_runs):
    gpu_id = i % n_gpus
    scripts[gpu_id] += f"python generate.py --target_df ../RNA_replicase_target.csv --gpu_id {gpu_id} --output_csv replicase_results_part_{i}.csv \n"

for i in range(n_gpus):
    with open(f"scripts_generate_replicase/run_generate_replicase_gpu_{i}.sh",'w') as f:
        f.write(scripts[i])

#write script to nohup all
with open("scripts_generate_replicase/run_all_nohup.sh",'w') as f:
    for i in range(n_gpus):
        f.write(f"nohup bash scripts_generate_replicase/run_generate_replicase_gpu_{i}.sh > generate_replicase_gpu_{i}.log &\n")