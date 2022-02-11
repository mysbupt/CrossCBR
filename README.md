# CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation


## Requirements
1. OS: Ubuntu 18.04 or higher version
2. python3.7.3
3. supported(tested) CUDA versions: 10.2
4. Pytorch1.8.0


## Code Structure
1. The entry script for training and evaluation is: train.py
2. The config file is: config.yaml
3. The script for data preprocess and dataloader: utility.py
4. The model folder: ./models.
5. The experimental logs in tensorboard-format are saved in ./runs.
6. The experimental logs in txt-format are saved in ./log.
7. The best model and associate config file for each experimental setting is saved in ./checkpoints.
