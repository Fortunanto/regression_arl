#!/usr/bin/zsh  
for i in {1..40}
do
echo -n "$i " >> errors_different_centers.csv
python train_baseline.py --seed $i >> errors_different_centers.csv
echo -n " " >> errors_different_centers.csv
python train_no_cqr.py --seed $i >> errors_different_centers.csv 
echo -n " " >> errors_different_centers.csv
python train.py --seed $i >> errors_different_centers.csv
echo -n "\n" >> errors_different_centers.csv
done