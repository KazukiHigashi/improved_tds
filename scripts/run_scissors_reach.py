"""Ordinary goal-reaching SAC, then collect only successful postures."""
import sys
from run_scissors_retention_v4 import main

if __name__=='__main__':
    main(['--version','simple','--output','runs/scissors_reach_20260923',*sys.argv[1:]])
