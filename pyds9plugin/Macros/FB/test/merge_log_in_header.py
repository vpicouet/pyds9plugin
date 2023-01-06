import re
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack, hstack
number_zeros = len(re.findall('[0-9]+',os.path.basename(filename))[0])

cat = Table.read(get(d, 'Path of the log catalog:', exit_=True))
path_dir = os.path.dirname(filename) #+ '/image_*.fits'

columns = cat.colnames[1:]
for line in cat:
    numbers = line['numbers']#.split()...
    length = len(str(numbers.split('-')[-1]))
    path = path_dir + '/image_%s[%s].fits'%('0'*(number_zeros-length),numbers)
    for file in globglob(path):
        for column in columns:
            fits.setval(file, column, value=line[column], comment="")
