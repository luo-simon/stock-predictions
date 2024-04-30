#!/bin/bash

models=("Linear" "ARIMA" "RandomForest")
stocks=("NVDA" "JPM" "HD" "UNH")

for m in "${models[@]}"
do
  for s in "${stocks[@]}"
  do
    python -m src.models.statistical.${m} --experiment_name ${m}_${s}_No-Feature-Engineering --stock $s --config configs/single_feature.yaml fit
    python -m src.models.statistical.${m} --experiment_name ${m}_${s}_Yes-Feature-Engineering --stock $s fit
  done
done

m="LSTM"

for s in "${stocks[@]}"
do
  python -m src.models.${m}.train fit --experiment_name ${m}_${s}_No-Feature-Engineering --stock $s --config configs/trainer_defaults.yaml --config configs/${m}.yaml --features log_return --model.input_dim 1
  python -m src.models.${m}.train fit --experiment_name ${m}_${s}_Yes-Feature-Engineering --stock $s --config configs/trainer_defaults.yaml --config configs/${m}.yaml
done

models=("CNN" "ConvLSTM")
for m in "${models[@]}"
do
  for s in "${stocks[@]}"
  do
    python -m src.models.${m}.train fit --experiment_name ${m}_${s}_No-Feature-Engineering --stock $s --config configs/trainer_defaults.yaml --config configs/${m}.yaml --features log_return --model.in_channels 1
    python -m src.models.${m}.train fit --experiment_name ${m}_${s}_Yes-Feature-Engineering --stock $s --config configs/trainer_defaults.yaml --config configs/${m}.yaml
  done
done
