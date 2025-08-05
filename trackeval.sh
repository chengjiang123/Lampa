#!/bin/bash

python3 tracking_trainer.py -m hmambav1 
python3 tracking_trainer.py -m hmambav2
python3 tracking_trainer.py -m hydra


python3 tracking_trainer.py -m hept 
python3 tracking_trainer.py -m mhaformer 
python3 tracking_trainer.py -m fullhybrid2 
python3 tracking_trainer.py -m fullfullhybrid2 

python3 tracking_trainer.py -m pemamba2 
python3 tracking_trainer.py -m gdlocal1 
python3 tracking_trainer.py -m fullmamba2 

python3 tracking_trainer.py -m lshgd
python3 tracking_trainer.py -m flatformer 
python3 tracking_trainer.py -m mamba 
