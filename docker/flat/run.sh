#! /bin/bash

# ALGOS=("loda", "lof", "ifor")
# TYPES=("type-b" "type-e" "type-r" "type-be" "type-br" "type-er" "type-ber")
# SOURCES=("huge" "complex" "simple" "single")

ALGOS=("ifor")
TYPES=("type-ber")
SOURCES=("simple")

OUTPUT_DIR=out
mkdir -p $OUTPUT_DIR

for TYPE in "${TYPES[@]}"
do
    DIR=data/data_parking/csv/$TYPE
    mkdir -p $DIR

    for SOURCE in "${SOURCES[@]}"
    do
        INPUT_FILE=$DIR/$SOURCE.$TYPE.csv
        for ALGO in "${ALGOS[@]}"
        do
            OUTPUT_FILE=$OUTPUT_DIR/$SOURCE.$TYPE.$ALGO.txt
            # rm $OUTPUT_FILE
            python run_simple.py $INPUT_FILE $ALGO > $OUTPUT_FILE
        done
    done
done

# for TYPE in "${TYPES[@]}"
# do
#     DIR=data/csv_normalized_hours/$TYPE
#     mkdir -p $DIR

#     for SOURCE in "${SOURCES[@]}"
#     do
#         rm $DIR/$SOURCE.$TYPE.csv
#         echo "Preparing $SOURCE $TYPE csv file"
#         node build/main.js \
#             --generator parking \
#             -t csv \
#             -f params_parking/$TYPE/params.$SOURCE.$TYPE.json \
#             -o $DIR/$SOURCE.$TYPE.csv \
#             --skip_timestamp --normalize_hours
#     done
# done

# echo "Printing final file sizes:"
# for FILE in $(find data_parking/csv -name '*.csv'); do
#     ANOMS=$(grep anomaly $FILE | wc -l)
#     LINES=$(wc -l $FILE)
#     echo "$FILE lines=$LINES anomalies=$ANOMS"
# done
# for FILE in $(find data_parking/csv_normalized_hours -name '*.csv'); do
#     ANOMS=$(grep anomaly $FILE | wc -l)
#     LINES=$(wc -l $FILE)
#     echo "$FILE lines=$LINES anomalies=$ANOMS"
# done

echo "Finished."
