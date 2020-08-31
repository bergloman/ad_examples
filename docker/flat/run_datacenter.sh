#! /bin/bash

# ALGOS=("loda" "lof" "ifor")
TYPES=("type1" "type2" "type3" "type12" "type13" "type23" "type123")
SOURCES=("huge" "complex" "simple" "single")
DATA_TYPES=("csv")

ALGOS=("loda")
# TYPES=("type1")
# SOURCES=("simple")
# DATA_TYPES=("csv")

OUTPUT_DIR=out/datacenter
mkdir -p $OUTPUT_DIR

for DATA_TYPE in "${DATA_TYPES[@]}"
do
    for TYPE in "${TYPES[@]}"
    do
        DIR=data/data_datacenter/$DATA_TYPE/$TYPE
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
