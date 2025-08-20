if [ -z "$ENTITYNET_DATA_DIR" ]; then
    echo "Error: ENTITYNET_DATA_DIR environment variable is not set"
    exit 1
fi

if [ ! -d "$ENTITYNET_DATA_DIR" ]; then
    echo "Error: ENTITYNET_DATA_DIR '$ENTITYNET_DATA_DIR' does not exist or is not a directory"
    exit 1
fi

if [ -d "$ENTITYNET_DATA_DIR/coco" ]; then
    echo "Error: Folder already exists: '$ENTITYNET_DATA_DIR/coco'"
    exit 1
fi

mkdir -p $ENTITYNET_DATA_DIR/coco/splits_karpathy
cd $ENTITYNET_DATA_DIR/coco/splits_karpathy
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json

mkdir -p $ENTITYNET_DATA_DIR/coco/images
cd $ENTITYNET_DATA_DIR/coco/images
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
