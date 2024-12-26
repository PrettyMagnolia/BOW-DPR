BASE_DIR=$(dirname "$PWD")
DATA_DIR=$BASE_DIR/ori_data
OUTPUT_DIR=$BASE_DIR/data

mkdir -p $OUTPUT_DIR

python prepare_data.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \