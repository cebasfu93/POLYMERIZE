NAME="PEP"

UNIT1="PEG"
N1="40"
UNIT2="PYR"
N2="1"
python polymerize.py MOL2/${UNIT1}.mol2 ${N1} MOL2/${UNIT2}.mol2 ${N2} ${NAME}.pdb

echo "source leaprc.gaff

loadamberparams FRCMOD/PEG.frcmod
loadamberparams FRCMOD/EAH.frcmod
loadamberparams FRCMOD/PYR.frcmod

loadoff LIB/PEG.lib
loadoff LIB/EAH.lib
loadoff LIB/PYR.lib

${NAME} = loadpdb ${NAME}.pdb
savemol2 ${NAME} ${NAME}.mol2 1

quit" > pdb2mol2.in

tleap -sf pdb2mol2.in
rm leap.log pdb2mol2.in
