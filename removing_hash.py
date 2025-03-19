# This program is a temporary solution to grabbing the data from the output files and removing the hash in the
#        beginning of the file to easily access the data for calculation 

import os

# Change the directory path HERE to the path that runs mcfacts 
HERE = '/Users/sray/Documents/1Saavik_Barry/test_mcfacts'
MCFACTS_RUNS_GAL = HERE + '/runs/gal0'
MCFACTS_RUNS = HERE + '/runs'

# Code that will match the number of objects and galaxies being run without the user having to change it manually
dir_path = (MCFACTS_RUNS_GAL)
objects = int((len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]) - 2)/3)
galaxies = len(next(os.walk('runs'))[1])

gal = [str(i).zfill(1) for i in range(galaxies)]
obj = [str(i) for i in range(objects)]

# Cycling through all the files to remove hash
for i in range(len(gal)):
    for j in range(len(obj)):
        old_file = '/Users/sray/Documents/1Saavik_Barry/test_mcfacts/runs/gal' + gal[i] + '/output_bh_binary_' + obj[j] + '.dat'
        with open(old_file, 'r') as files:
            lines = files.readlines()

        if lines:
            lines[0] = lines[0][2:]

        new_file = '/Users/sray/Documents/1Saavik_Barry/test_mcfacts/runs/gal' + gal[i] + '/output_bh_binary_' + obj[j] + '.dat'
        with open(new_file, 'w') as files:
            files.writelines(lines)