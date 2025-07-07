README file for bechmark sets
-----------------------------

This is a collection of multiple alignment benchmarks in a uniform 
format that is convenient for further analysis. All files are in 
FASTA format, with upper-case letters used to indicate aligned 
columns. 

See References below for original sources of benchmark data.

Benchmarks are:

bali2dna
BALIBASE v2, reverse-translated to DNA

bali2dnaf
Bali2dbn, with frame-shifts induced by random insertions of one
or two nucleotides into the middle 50% of exactly one sequence
in each set.

bali3
BALIBASE v3.

bali3pdb
BALIS, the structural subset of BALIBASE v3.

bali3pdbm
MU-BALIS, i.e. BALIS re-aligned by MUSTANG.

ox
OXBENCH.

oxm
MU-OXBENCH, i.e. OXBENCH re-aligned by MUSTANG.

oxx
OXBENCH-X, i.e. the Extended set in OBENCH.

prefab4
PREFAB v4.

prefab4ref
PREFAB-R, i.e. the pair-wise reference pairs in PREFAB v4.

prefab4refm
MU-PREFAB-R, i.e. PREFAB-R re-aligned by MUSTANG.

sabre
Consistent multiple alignments constructed from SABMARK v1.65.

sabrem
MU-SABRE, i.e. SABRE re-aligned by MUSTANG.

Directory structure under each benchmark is:

in/
Input sequences.

ref/
Reference alignments. Upper-case regions indicate conservative 
regions that are intended for use in assessment. Lower-case regions 
should not be used.

info/
Contains ids.txt (list of set identifiers that are filenames in ref/ 
and in/), nrseqs.txt (number of sequences in each set), and 
pctids.txt (%id in conservative regions in each set).

qscore/
Output from qscore for each set for selected methods. Methods 
include:

clw
CLUSTALW.

dialignt
DIALIGN-TX v1.0.2.

fsa
FSA v1.14.5.

fsamaxsn
FSA with -maxsn option.

kalign2
KALIGN v2.03.

mafft6.603einsi
mafft6.603ginsi
mafft6.603linsi
MAFFT v6.603 using EINSI, GINSI and LINSI.

mummals
MUMMALS dated 8/2/08.

mus4
MUSCLE v4.0.

pc1.12
PROBCONS v1.12

prankfclwtree
PRANK +F with CLUSTALW tree.

probalign
PROBALIGN v1.0.

Alignment accuracy scoring.
===========================
Alignments were scored using qscore:

	http://www.drive5.com/qscore

The command-line used to produce most of the qscore files is:

qscore -seqdiffwarn -ignoremissingseqs -ignoretestcase 
  -test <testfile> -ref <reffile> -truncname -quiet

In some case post-processing of alignments is necessary. For example, 
DIALIGN-TX combines aligned (upper-case) and non-aligned (lower-case) 
letters in a single column, which qscore does not support. This is 
fixed by splitting a mixed-case column into two columns, one upper-
case and one lower-case. To exclude the lower-case columns from 
assessment, the -ignoretestcase option of qscore should /not/ be 
used.

References
----------

Thompson JD, Koehl P, Ripp R, Poch O (2005) BAliBASE 3.0: latest 
developments of the multiple sequence alignment benchmark. Proteins 
61: 127-136.

Bahr A, Thompson JD, Thierry JC, Poch O (2001) BAliBASE (Benchmark 
Alignment dataBASE): enhancements for repeats, transmembrane 
sequences and circular permutations. Nucleic Acids Res 29: 323-326.

Thompson JD, Plewniak F, Poch O (1999) BAliBASE: a benchmark 
alignment database for the evaluation of multiple alignment programs. 
Bioinformatics 15: 87-88.

Van Walle I, Lasters I, Wyns L (2005) SABmark--a benchmark for 
sequence alignment that covers the entire known fold space. 
Bioinformatics 21: 1267-1268.

Raghava GP, Searle SM, Audley PC, Barber JD, Barton GJ (2003) 
OXBench: a benchmark for evaluation of protein multiple sequence 
alignment accuracy. BMC Bioinformatics 4: 47.

Edgar RC (2004) MUSCLE: multiple sequence alignment with high 
accuracy and high throughput. Nucleic Acids Res 32: 1792-1797.

