"""v5 launch entry point; only Scissors 3--5, fresh output, same 3M budget."""
import sys
from run_scissors_retention_v4 import main

if __name__=='__main__':
    main(['--version','v5','--output','runs/scissors_retention_v5_20260921',*sys.argv[1:]])
