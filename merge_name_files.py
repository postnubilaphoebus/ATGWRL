# files are taken from here
# https://www.ssa.gov/oact/babynames/limits.html
import os

filenames = ["names/yob" + str(x) + ".txt" for x in range(1880, 2022, 1)]
with open('unfiltered_names.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

with open("unfiltered_names.txt", "r") as f, open("names.txt", "w") as g:
    while True:
        line = f.readline()
        sep = ','
        stripped = line.split(sep, 1)[0]
        if not stripped:
            break
        g.write(stripped + "\n")

os.remove("unfiltered_names.txt")
