import os
import sys

res = os.system('qrun -b')
if not res:
    exit(0)
for i in range(10):
    os.system('qrun %s 8000' % sys.argv[1])
