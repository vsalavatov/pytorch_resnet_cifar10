#!/bin/bash

echo "Preparing environment..."

printf "0 1\n1 2\n2 0" > topology.txt

echo "Starting master..."
python3.7 consensus_master.py --master-port 8999 --topology-path topology.txt &

echo Common checkpoint path is $CHECKPOINT_PATH

echo "Starting agent 0 (init leader)"
python3.7 lsr_consensus_trainer.py -t 0 --enable-log --init-leader --total-agents 3 --master-port 8999 --agent-port 11000 --save-dir "$CHECKPOINT_PATH" &

for token in {1..2}; do
  echo Starting agent $token
  python3.7 lsr_consensus_trainer.py -t $token --total-agents 3 --master-port 8999 --agent-port 1100$token --save-dir "$CHECKPOINT_PATH" &
done