#!/bin/bash

SCRIPTS=(ds1.sh ds2.sh ds3.sh ds4.sh ds5.sh)
prev_jid=""

for script in "${SCRIPTS[@]}"; do
    if [ -z "$prev_jid" ]; then
        jid=$(sbatch "$script" | awk '{print $4}')
    else
        jid=$(sbatch --dependency=afterok:$prev_jid "$script" | awk '{print $4}')
    fi
    echo "Submitted $script as Job $jid"
    prev_jid=$jid
done