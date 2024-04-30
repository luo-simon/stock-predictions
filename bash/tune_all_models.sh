#!/bin/bash

stocks=("NVDA" "JPM" "HD" "UNH")

for s in "${stocks[@]}"
do
  python -m src.models.statistical.Linear --experiment_name Linear_${s} --stock $s fit
done

models=("ARIMA" "RandomForest")

for m in "${models[@]}"
do
  for s in "${stocks[@]}"
  do
    python -m src.models.statistical.${m} --experiment_name ${m}_${s} --stock $s fit
    python -m src.models.statistical.${m} --experiment_name ${m}_${s} --stock $s tune 9
  done
done

models=("LSTM" "CNN" "ConvLSTM")
for m in "${models[@]}"
do
  for s in "${stocks[@]}"
  do
    python -m src.models.${m}.train fit --experiment_name ${m}_${s} --stock $s --config configs/trainer_defaults.yaml --config configs/${m}.yaml
    python -m src.models.${m}.train tune --experiment_name ${m}_${s} --stock $s --config configs/trainer_defaults.yaml --config configs/${m}.yaml --n_trials 9
  done
done
