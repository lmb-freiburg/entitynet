#!/usr/bin/env bash

if [ -z "$ENTITYNET_DATA_DIR" ]; then
    echo "Error: ENTITYNET_DATA_DIR environment variable is not set"
    exit 1
fi

if [ ! -d "$ENTITYNET_DATA_DIR" ]; then
    echo "Error: ENTITYNET_DATA_DIR '$ENTITYNET_DATA_DIR' does not exist or is not a directory"
    exit 1
fi

if [ -d "$ENTITYNET_DATA_DIR/iNat" ]; then
    echo "Error: Folder already exists: '$ENTITYNET_DATA_DIR/iNat'"
    exit 1
fi

# source: https://github.com/visipedia/inat_comp
mkdir -p $ENTITYNET_DATA_DIR/iNat/2019
cd $ENTITYNET_DATA_DIR/iNat/2019
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train2019.json.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2019/val2019.json.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2019/categories.json.tar.gz
tar -xf train_val2019.tar.gz
tar -xf train2019.json.tar.gz
tar -xf val2019.json.tar.gz
tar -xf categories.json.tar.gz

mkdir -p $ENTITYNET_DATA_DIR/iNat/2021
cd $ENTITYNET_DATA_DIR/iNat/2021
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz
tar -xf train.tar.gz
tar -xf train.json.tar.gz
tar -xf val.tar.gz
tar -xf val.json.tar.gz