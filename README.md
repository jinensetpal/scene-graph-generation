# Scene Graph Generation Challenge
DagsHub x ML@Purdue Hackathon

## Setup

From the root of the repository:
```
pip install -r requirements.txt  # please use a virtual environment!
aws s3 cp -r s3://visualgenome data
```

## Inference Pipeline
```
python -m src.model.arch
```

## Project Structure
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
    └── utils <- helper scripts
        ├── pipeline.sh     <- run all the code at once
        └── colab_setup.sh  <- automate colab setup

-----
