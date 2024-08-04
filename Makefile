# Declarations
.PHONY: all clean

all: clean tests plots #vera_plots
tests: mcfacts_sim

######## Definitions ########
#### Package ####
# Version number for the repository
# This is where you change the version number by hand. Not anywhere else.
# Alpha begins at 0.1.0
# Feature-complete Alpha begins at 0.2.0
# Beta begins at 0.3.0
VERSION=0.0.0

### Should work for everyone ###
# Current directory
#HERE=$(shell pwd)
HERE=./

#### Scripts ####
MCFACTS_SIM_EXE = ${HERE}/scripts/mcfacts_sim.py
POPULATION_PLOTS_EXE = ${HERE}/scripts/population_plots.py
VERA_PLOTS_EXE = ${HERE}/scripts/vera_plots.py
MSTAR_RUNS_EXE = ${HERE}/scripts/vera_mstar_bins.py
MSTAR_PLOT_EXE = ${HERE}/src/mcfacts/outputs/plot_mcfacts_handler_quantities.py

#### Setup ####
HC_EXP_NAME = retro_binaries
HC_RUN_NAME = sg_fret0p1
HC_WKDIR = ${HERE}../mcfacts_research/paper2_qXeff/${HC_EXP_NAME}/${HC_RUN_NAME}/
HC_INPUT_FILE = ${HERE}recipes/paper2/${HC_EXP_NAME}/paper2_${HC_RUN_NAME}.ini

MSTAR_RUNS_WKDIR = ${HERE}/runs_mstar_bins
# NAL files might not exist unless you download them from
# https://gitlab.com/xevra/nal-data
# scripts that use NAL files might not work unless you install
# gwalk (pip3 install gwalk)
FNAME_GWTC2_NAL = ${HOME}/Repos/nal-data/GWTC-2.nal.hdf5
wd=${HC_WKDIR}#$(shell pwd)/test_output

######## Instructions ########
#### Install ####

version: clean
	echo "__version__ = '${VERSION}'" > __version__.py
	echo "__version__ = '${VERSION}'" > src/mcfacts/__version__.py

install: clean version
	pip install -e .

#### Test one thing at a time ####

mcfacts_sim: clean
	python ${MCFACTS_SIM_EXE} \
		--fname-log out.log --work-directory ${wd} \
		--seed 3456789012

plots:  #mcfacts_sim
	python ${POPULATION_PLOTS_EXE} --fname-mergers ${wd}/output_mergers_population.dat --plots-directory ${wd}

vera_plots: mcfacts_sim
	python3 ${VERA_PLOTS_EXE} \
		--cdf chi_eff chi_p M gen1 gen2 t_merge \
		--verbose

mstar_runs:
	echo ${MSTAR_RUNS_WKDIR}/early
	echo ${MSTAR_RUNS_WKDIR}/late
#	python3 ${MSTAR_RUNS_EXE} \
#		--number_of_timesteps 100 \
#		--n_iterations 10 \
#		--dynamics \
#		--feedback \
#		--mstar-min 1e9 \
#		--mstar-max 1e13 \
#		--nbins 9 \
#        --scrub \
#		--fname-nal ${FNAME_GWTC2_NAL} \
#		--wkdir ${MSTAR_RUNS_WKDIR}
#	python3 ${MSTAR_PLOT_EXE} --run-directory ${MSTAR_RUNS_WKDIR}/early
#	python3 ${MSTAR_PLOT_EXE} --run-directory ${MSTAR_RUNS_WKDIR}/late

qxeff:
#	python3 ${MCFACTS_SIM_EXE} \
#		--fname-log out.log \
#		--fname-ini=${HC_INPUT_FILE}  \
#		--work-directory=${wd}
	python3 ${POPULATION_PLOTS_EXE} \
	    --plots-directory=${wd} \
		--fname-mergers=${wd}/output_mergers_population.dat

#### CLEAN ####
clean:
	rm -rf ${wd}/run*
	rm -rf ${wd}/output_mergers_population.dat
	rm -rf ${wd}/output_mergers_emris.dat
	rm -rf ${wd}/m1m2.png
	rm -rf ${wd}/merger_mass_v_radius.png
	rm -rf ${wd}/q_chi_eff.png
	rm -rf ${wd}/time_of_merger.png
	rm -rf ${wd}/merger_remnant_mass.png
	rm -rf ${wd}/gw_strain.png
	rm -rf ${wd}/r_chi_p.png
	rm -rf ${wd}/out.log
	rm -rf ${wd}/output_mergers_lvk.dat
	rm -rf ${wd}/output_mergers_survivors.dat
