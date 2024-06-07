START_TIME=`date +%s`
data_partition="noniid"
iternum=200
client=3
beta=0.1
dataset="eurosat"
model="resnet20_cifar"
sample_fraction=1.0
load_path=None
dir_path=./log/${dataset}_${data_partition}_${model}_beta${beta}_it${iternum}_c${client}_p${sample_fraction}
lr=0.01
round=100


cd ..
mkdir $dir_path

alg="fedavg"
nohup python -u main.py --n_round $round --lr $lr --alg $alg --dataset $dataset --gpu "4" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/${alg}_${START_TIME}.log &

# alg="fedprox"
# mu=0.01
# nohup python -u main.py --alg $alg --mu $mu --dataset $dataset --gpu "4" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/${alg}_${mu}_${START_TIME}.log &

# alg="scaffold"
# nohup python -u main.py --alg $alg --dataset $dataset --gpu "4" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/${alg}_${START_TIME}.log &


# ========== FedGC =========
# fedgc_num_gen=10000
# fedgc_allocation="inverse"
# fedgc_change_aggr=0

# alg="fedavg"
# nohup python -u main.py --gpu "2" --fedgc --fedgc_num_gen $fedgc_num_gen --fedgc_allocation $fedgc_allocation --fedgc_change_aggr $fedgc_change_aggr --alg $alg --dataset $dataset --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/fedgc_${alg}_gen${fedgc_num_gen}_${fedgc_allocation}_agg${fedgc_change_aggr}_${START_TIME}.log &

# alg="fedprox"
# mu=0.01
# nohup python -u main.py --gpu "4" --fedgc --fedgc_num_gen $fedgc_num_gen --fedgc_allocation $fedgc_allocation --fedgc_change_aggr $fedgc_change_aggr --alg $alg --mu $mu --dataset $dataset --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/fedgc_${alg}_${mu}_gen${fedgc_num_gen}_${fedgc_allocation}_agg${fedgc_change_aggr}_${START_TIME}.log &

# alg="scaffold"
# nohup python -u main.py --gpu "5" --fedgc --fedgc_num_gen $fedgc_num_gen --fedgc_allocation $fedgc_allocation --fedgc_change_aggr $fedgc_change_aggr --alg $alg --dataset $dataset --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/fedgc_${alg}_gen${fedgc_num_gen}_${fedgc_allocation}_agg${fedgc_change_aggr}_${START_TIME}.log &

# alg="scaffold"
# python main.py --fedgc --alg $alg --dataset $dataset --gpu "2" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta 