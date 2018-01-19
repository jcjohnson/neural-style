#!/bin/bash
if [ $# -eq 1 ]; 
	then
    a=`find $1 -type f`
else 
    a=`find $1 -type f -name "*.$2"`
fi
b=$(echo "$a" | tr '\n' ,)
b=${b::-1}
echo $b
