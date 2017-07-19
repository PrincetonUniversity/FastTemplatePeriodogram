"""
Reads `<package>/version.py`, and replaces all instances
of `${UPPER_CASE_VARIABLE}` in files with the `.in` suffix
with their corresponding values. Useful for, e.g., keeping
the version up-to-date in the README.
"""
import os
import glob
import sys
import re

verbose = len(sys.argv) > 1 and sys.argv[1] == '-v'

package = 'ftperiodogram'


if verbose:
    print("reading version.py")
with open(os.path.join(package, 'version.py'), 'r') as f:
    exec(f.read())

if verbose:
    print("generating dictionary")
variables = {lvar : globals()[lvar] for lvar in locals().keys()
             if lvar.isupper()}

if verbose:
    print("vars:")
    for var in variables.keys():
        print("  %s"%var)

# get list of .in files
infiles = glob.glob('*.in')

if verbose:
    print("infiles:")
    for infile in infiles:
        print("  %s"%infile)

if verbose:
    print("replacements:")
# replace variables in each file
for infile in infiles:

    ofile = infile.replace('.in', '')
    with open(infile, 'r') as f:
        # read text
        txt = f.read()

        ninstances_total = 0
        for variable in variables.keys():

            key = "${%s}"%(variable)
            value = str(variables[variable])
            regex_key = re.escape(key)

            ninstances = len([m.start()
                              for m in re.finditer(regex_key, txt)])

            if verbose and ninstances > 0:
                print(" ".join(["  [%s] replacing %d"%(infile, ninstances),
                                "instances of '%s' with '%s'"%(key, value)]))

            txt = txt.replace(key, value)

        # write replaced text
        with open(ofile, 'w') as fo:
            fo.write(txt)
