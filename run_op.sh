# run matmul

num_threads=("16")
n=("256" "512" "1024" "2048" "4096" "8192")

for nt in "${num_threads[@]}"
do
    for N in "${n[@]}"
    do
        OMP_NUM_THREADS=${nt} \
        KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 KMP_SETTINGS=1 python matmul_bench.py --N ${N} 1>matmul_N${N}_t${nt}.log 2>&1 
        echo "Finish matrix size=${N} threads=${nt}. Sleep..."
        sleep 5
    done
done

## run conv
#
##num_threads=("1" "2" "4" "8" "16" "24")
#num_threads=("16")
#
#for nt in "${num_threads[@]}"
#do
#
#        OMP_NUM_THREADS=${nt} \
#        KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 KMP_SETTINGS=1 python conv_bench.py 1>conv_t${nt}.log 2>&1 
#        echo "Finish threads=${nt}. Sleep..."
#        sleep 5
#done
