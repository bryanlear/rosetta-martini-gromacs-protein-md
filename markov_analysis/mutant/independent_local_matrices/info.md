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

- step 7:

`cd /independent_local_matrices && ls -l msm_block{1,2,3}.xtc msm_block{1,2,3}/rg.npy tica/msm_block{1,2,3}_tica.npy transition_matrices/msm_block{1,2,3}_labels_scc.npy stationary_distribution/msm_block{1,2,3}_pi_rev_tau10.npy stationary_distribution/msm_block{1,2,3}_scc_local_map.npy ../reference/conf_replica_00.gro 2>&1`