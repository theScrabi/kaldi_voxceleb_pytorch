#!/bin/bash

## creates a file that displays the time of every wave entry

if [ $# -lt 1 ]; then
    echo "You nee to enter the wav.scp file you"
    echo "want to calculate the duration over."
    exit 1
fi

wav_scp=$1

while IFS=' ' read -ra line
do
    id=${line[0]}
    file=${line[1]}
    echo $id $(soxi -D $file)
done < $wav_scp
