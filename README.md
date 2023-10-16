# Scene Graph Generation Challenge
DagsHub x ML@Purdue Hackathon

This is the template repository for the Scene Graph Generation Challenge.

## Setup

From the root of the repository:
```bash
pip install -r requirements.txt  # please use a virtual environment!
aws s3 cp s3://visualgenome data --recursive
```

### Forking the template to your repository
```bash
git remote add base https://dagshub.com/ML-Purdue/sgg-template.git
git pull base --rebase
git branch -M main
git push origin main
```

### Updating the Fork
```bash
git pull base main --allow-unrelated-histories
```
Then, resolve any merge conflicts, commit and push!

## Inference Pipeline
```bash
python -m src.model.arch
```

## Project Structure
```
├── LICENSE
├── README.md
├── requirements.txt
├── data                    <- visual genome dataset
└── src
    ├── const.py            <- constants (hyperparameters, paths)
    ├── data
    │   └── visualgenome.py <- dataset sampler
    ├── model
    │   ├── arch.py         <- model architecture
    │   ├── agcn.py         <- attention graph convolutional network
    │   └── repn.py         <- relationship proposal network
    └── utils.py            <- helper functions (nms)
```

-----
