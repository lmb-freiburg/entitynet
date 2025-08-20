#!/usr/bin/env bash

if [ -z "$ENTITYNET_DATA_DIR" ]; then
    echo "Error: ENTITYNET_DATA_DIR environment variable is not set"
    exit 1
fi

if [ ! -d "$ENTITYNET_DATA_DIR" ]; then
    echo "Error: ENTITYNET_DATA_DIR '$ENTITYNET_DATA_DIR' does not exist or is not a directory"
    exit 1
fi

if [ -d "$ENTITYNET_DATA_DIR/cub200" ]; then
    echo "Error: Folder already exists: '$ENTITYNET_DATA_DIR/cub200'"
    exit 1
fi

# source: https://github.com/visipedia/inat_comp
mkdir -p $ENTITYNET_DATA_DIR/cub200
cd $ENTITYNET_DATA_DIR/cub200
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
mv CUB_200_2011.tgz?download=1 CUB_200_2011.tgz
tar -xf CUB_200_2011.tgz