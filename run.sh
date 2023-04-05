#!/bin/bash
for i in {0..31}
do
	taskset --cpu-list $i python3 vdb.py $i &
done
