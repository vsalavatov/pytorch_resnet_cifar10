#!/bin/bash

echo "Preparing environment..."

echo "0 1\
1 2\
2 3" > topology.txt

echo "Starting master..."
python3.8 consensus_master.py --master_host 0.0.0.0 --master_port 8999 --topology-path topology.txt &

echo 'Common checkpoint path is $CHECKPOINT_PATH'

echo "Starting agent 0 (init leader)"
python3.8 plain_consensus_trainer.py -t 0 --init-leader --master_port 8999 --agent-port 9000 --save-dir $CHECKPOINT_PATH &

for token in {1..2}; do
  echo "Starting agent $token"
  python3.8 plain_consensus_trainer.py -t $token --master_port 8999 --agent-port 900$token --save-dir $CHECKPOINT_PATH &
done