#! /bin/bash

#ALGOS=("loda" "lof" "ifor")
# TYPES=("type-b" "type-e" "type-r" "type-be" "type-br" "type-er" "type-ber")
# SOURCES=("huge" "complex" "simple" "single")
DATA_TYPES=("csv" "csv_normalized_hours")

ALGOS=("ifor" "lof")
TYPES=("type-e" "type-ber")
SOURCES=("simple")
# DATA_TYPES=("csv_normalized_hours")

OUTPUT_DIR=out
mkdir -p $OUTPUT_DIR

for DATA_TYPE in "${DATA_TYPES[@]}"
do
    for TYPE in "${TYPES[@]}"
    do
        DIR=data/data_parking/$DATA_TYPE/$TYPE
        mkdir -p $DIR
        PIDS=""

        for SOURCE in "${SOURCES[@]}"
        do
            INPUT_FILE=$DIR/$SOURCE.$TYPE.csv
            for ALGO in "${ALGOS[@]}"
            do
                OUTPUT_FILE=$OUTPUT_DIR/$DATA_TYPE.$SOURCE.$TYPE.$ALGO.txt
                # rm $OUTPUT_FILE
                python run_simple.py $INPUT_FILE $ALGO > $OUTPUT_FILE &
                PIDS="$PIDS $!"
            done
        done
        echo "Waiting for batch of child processes: $PIDS"
        wait $PIDS
        echo "Batch of child processes finished ($DATA_TYPE $TYPE $SOURCE)"
    done
done

echo "Finished."
