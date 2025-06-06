#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE=$SCRIPT_DIR
RECEPTORID=6seo
SEQ='GPRSVASSKLWMLEFSAFLEQQQDPDTYNKHLFVHIGQSSPSYSDPYLEAVDIRQIYDKFPEKKGGLKDLFERGPSNAFFLVKFWADLNTNIEDEGSSFYGVSSQYESPENMIITCSTKVCSFGKQVVEKVETEYARYENGHYSYRIHRSPLCEYMINFIHKLKHLPEKYMMNSVLENFTILQVVTNRDTQETLLCIAYVFEVSASEHGAQHHIYRLVKE'   #6seo
POCKET='40,41,42,43,46,47,74,76,162,185,187,189,190,191,196,197,198,200'   #6seo
PATH_DIR=$BASE/$RECEPTORID/6seo_16
PEPTIDELENGTH=16
DATADIR=$PATH_DIR/result
NITER=300
JUMPOUT=30

python3 $BASE/design.py \
		--receptor_seq=$SEQ \
		--receptor_if_residues=$POCKET \
		--peptide_length=$PEPTIDELENGTH \
		--output_dir=$DATADIR \
		--num_iterations=$NITER \
        --plDDT_only=True \
        --receptor_name=$RECEPTORID \
		--jumpout_num=$JUMPOUT \
