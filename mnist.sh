#python main.py --training_epochs 2 --layer_features 256 10

for i in 1 2
do
  for j in 64 128
  do
    python main.py --training_epochs $i --layer_features $j 10
  done
done