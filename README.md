# HighPlay： Cyclic Peptide Sequence Design Based on Reinforcement Learning and Protein Structure Prediction

This is the code for "HighPlay： Cyclic Peptide Sequence Design Based on Reinforcement Learning and Protein Structure Prediction" paper.

## Code Hierarchy

```shell
bashCopy code/
├── data/                    
│   └── target.csv           
├── design.py                
├── design.sh                
├── mcts.py                  
├── mutate.py                
├── policyvaluenet.py        
├── pre.py                  
├── requirements.txt       
├── train.py   
├── LICENSE        
└── README.md       
```

## Instructions

### Clone this repository and cd into it.

```shell
mkdir your_workspace
cd your_workspace
git clone https://github.com/htianlin/HighPlay
```


### Install dependency

Running this  command for installing dependency in docker:

```shell
pip install requirments.txt
```


### Download parameters

Download alphafold params from [AlphaFold Github](https://github.com/deepmind/alphafold/blob/main/scripts/download_alphafold_params.sh)

```shell
bash download_alphafold_params.sh ./
```

### Using HighPlay model

It supports optimization of cyclic peptide design with input of only the target protein sequence.

You can run it in your local conda environment

```shell
source activate highplay
bash ./design.sh
```

## Support or Report Issues

If you encounter any issues or need support while using HighPlay, please report the issue in the [GitHub Issues](https://github.com/your_username/HighPlay/issues) .

## Copyright and License

This project is governed by the terms of the MIT License. Prior to utilization, kindly review the LICENSE document for comprehensive details and compliance instructions.




