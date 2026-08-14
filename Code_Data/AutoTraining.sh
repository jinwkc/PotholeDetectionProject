#!/bin/bash

python3 $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
darknet detector train $9 $4 $3 -clear

#./test.sh 
# $1 = Python File
# $2 img data Dir
# $3 nn_last.weights path
# $4 nn.cfg path   
# $5 original yolo.names path
# $6 python save new Train Data Save Dir Path
# $7 new train data list txt file save path
# $8 new validation data list txt file save path
# $9 new yolo.data path 
# $10 new Yolo Model backup dir