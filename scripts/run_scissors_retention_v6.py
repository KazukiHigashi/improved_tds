"""v6: shared reach/hold correction; fresh Scissors 3--5, 3M steps each."""
import sys
from run_scissors_retention_v4 import main

if __name__ == '__main__':
    main(['--version', 'v6', '--output', 'runs/scissors_retention_v6_20260921', *sys.argv[1:]])
