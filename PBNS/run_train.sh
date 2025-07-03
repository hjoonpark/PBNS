#!/bin/bash

set -e

GPU=0
NAME=output-pbns
OBJECT=Outfit
BODY=Body

echo "=============================="
echo "     Starting Training        "
echo "=============================="
python train.py -g "$GPU" -b "$BODY" -o "$OBJECT" -n "$BODY"

echo "=============================="
echo "Ended Training"
echo "=============================="