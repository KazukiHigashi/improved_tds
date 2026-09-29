"""v7 shared stability/phase-aware SAC, only Scissors 3--5, 3M per task."""
import sys
from run_scissors_retention_v4 import main

if __name__=='__main__':
    main(['--version','v7','--output','runs/scissors_retention_v7_20260921',*sys.argv[1:]])
