for SEED in 0 1 2 3 4 5 6 7 8 9
  do
    python main.py --policy GFN --N 1 --env $1 --seed $SEED
    python main.py --policy DDPG --N 1 --env $1 --seed $SEED
  done