#!/bin/bash

set -e

GPU=0
NAME=train
OBJECT=Outfit
BODY=Body

echo "=============================="
echo "     Starting Training        "
echo "=============================="
# python train.py -g "$GPU" -b "$BODY" -o "$OBJECT" -n "$NAME"

echo "=============================="
echo "     Ended Training"
echo "=============================="


echo "=============================="
echo "     Starting Rendering       "
echo "=============================="
python render_trained.py

echo "=============================="
echo "     Ended Training"
echo "=============================="