- To run step 2:

`conda activate bio_env && cd /Users/bry_lee/Desktop/protein_simulation/MUTANT_extended_simulation/mutant_analysis_pkg/independent_local_matrices && python step2_tica.py --lag 10 --var 0.95 2>&1 | tee tica_run.log`

- Step 3:

`conda activate bio_env && python step3_kmedoids.py --k 200 --scan 2>&1 | tee clustering_run.log`

- step 4:

`python step4_count_matrices.py --lag 10 2>&1 | tee count_matrices_run.log`

- step 5:

`conda activate bio_env && python step5_transition_matrices.py --lag 10 2>&1 | tee transition_matrices_run.log`

- step 6:

`python step6_stationary_distribution.py --lag 10 2>&1 | tee stationary_dist_run.log`