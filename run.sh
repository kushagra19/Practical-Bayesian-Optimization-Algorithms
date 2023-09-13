#!/bin/bash

mpirun -n 64 --mca btl_tcp_if_include eno1,ib0  /apps/parallel_studio_xe_2020_update2_cluster_edition/intelpython3/bin/python main.py -e ZOBO -ed $4 -obj_fn $1 -m $2 -d $3 -zacq $5 -o 0 -b $6 -r $7 -nm $8 -nv $9 -nr ${10} -rs ${11} -s ${12} -i ${13}
mpirun -n 64 --mca btl_tcp_if_include eno1,ib0  /apps/parallel_studio_xe_2020_update2_cluster_edition/intelpython3/bin/python main.py -e Prabu_FOBO -ed $4 -obj_fn $1 -m $2 -d $3 -zacq $5 -o 1 -q convex -gacq PrabuAcquistionFunction -b $6 -r $7 -nm $8 -nv $9 -nr ${10} -rs ${11} -s ${12} -i ${13}
mpirun -n 64 --mca btl_tcp_if_include eno1,ib0  /apps/parallel_studio_xe_2020_update2_cluster_edition/intelpython3/bin/python main.py -e topk -ed $4 -obj_fn $1 -m $2 -d $3 -zacq $5 -o 1 -q topk -gacq SumGradientAcquisitionFunction -b $6 -r $7 -nm $8 -nv $9 -nr ${10} -rs ${11} -s ${12} -i ${13}
mpirun -n 64 --mca btl_tcp_if_include eno1,ib0  /apps/parallel_studio_xe_2020_update2_cluster_edition/intelpython3/bin/python main.py -e topk_convex -ed $4 -obj_fn $1 -m $2 -d $3 -zacq $5 -o 1 -q topk_convex -gacq SumGradientAcquisitionFunction -b $6 -r $7 -nm $8 -nv $9 -nr ${10} -rs ${11} -s ${12} -i ${13}
