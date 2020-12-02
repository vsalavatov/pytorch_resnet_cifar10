#!/bin/bash

echo "Preparing environment..."

printf "0 1\n1 2\n2 3\n3 4\n4 5\n5 6\n6 7\n7 8\n8 9\n9 0\n0 5\n1 6\n2 7\n3 8\n4 9" > topology.txt

echo "Starting master..."
python3.7 consensus_master.py --master-port 8999 --topology-path topology.txt &

echo Common checkpoint path is $CHECKPOINT_PATH

echo "Starting agent 0 (init leader)"
python3.7 lsr_consensus_trainer.py --target-split -t 0 --enable-log --init-leader --total-agents 10 --master-port 8999 --agent-port 11000 --save-dir "$CHECKPOINT_PATH" &

for token in {1..9}; do
  echo Starting agent $token
  python3.7 lsr_consensus_trainer.py --target-split -t $token --total-agents 10 --master-port 8999 --agent-port 1100$token --save-dir "$CHECKPOINT_PATH" &
done