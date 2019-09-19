##step 1
pip install tensor2tensor
##step 2
t2t-trainer --registry_help

PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=$HOME/data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/data/t2t_train/$PROBLEM/$MODEL-$HPARAMS
#cpu_threads="${cpu_threads:-1}"

echo ${OMP_NUM_THREADS}

#mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

## step 3, Generate data
#t2t-datagen \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM
#step 4
KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 \
KMP_SETTINGS=1 t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --worker_gpu=0 \
  --hparams='batch_size=1024' \
  --intra_op_parallelism_threads 1 \
  --inter_op_parallelism_threads 1 \
  --log_step_count_steps 10

