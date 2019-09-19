#OMP_NUM_THREADS=1 nohup ./t2t.sh 1>t2t_t1.log 2>&1 &
#echo "sleep..."
#sleep 1600
#killall t2t-trainer
#
#OMP_NUM_THREADS=2 nohup ./t2t.sh 1>t2t_t2.log 2>&1 &
#echo "sleep..."
#sleep 800
#killall t2t-trainer
#
#OMP_NUM_THREADS=4 nohup ./t2t.sh 1>t2t_t4.log 2>&1 &
#echo "sleep..."
#sleep 400
#killall t2t-trainer
#
#OMP_NUM_THREADS=8 nohup ./t2t.sh 1>t2t_t8.log 2>&1 &
#echo "sleep..."
#sleep 400
#killall t2t-trainer
#
#OMP_NUM_THREADS=16 nohup ./t2t.sh 1>t2t_t16.log 2>&1 &
#echo "sleep..."
#sleep 400
#killall t2t-trainer
#
#OMP_NUM_THREADS=24 nohup ./t2t.sh 1>t2t_t24.log 2>&1 &
#echo "sleep..."
#sleep 400
#killall t2t-trainer

OMP_NUM_THREADS=12 nohup ./t2t.sh 1>t2t_t12.log 2>&1 &
echo "sleep..."
sleep 400
killall t2t-trainer

