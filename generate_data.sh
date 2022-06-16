# Generate lattice ising data
for DIM in 25
do
  for SIGMA in .25
    do
      python pcd.py \
            --save_dir ./DATASETS/ising_dim_${DIM}_sigma_${SIGMA} \
            --model lattice_ising --data_model lattice_ising --dim ${DIM} --sigma ${SIGMA} &
  done
done
