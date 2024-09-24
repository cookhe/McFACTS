<h1 align="center">
    <br>
    <a href="https://github.com/bmckernan/test_mcfacts"><img src="branding/logo/mcfacts_logo.png" alt="Markdownify" width="500"></a>
    <br>
    <span style="font-weight: normal">
        <b>M</b>onte <b>c</b>arlo <b>F</b>or <b>A</b>GN <b>C</b>hannel <b>T</b>esting and <b>S</b>imulations
    </span>  
    <br>
</h1>

<h4 align="center">A python package that does the AGN channel for you!</h4>

### Installation

To clone and run this application, you'll need [Git](https://git-scm.com) and [Conda](https://docs.conda.io/en/latest/).

The latest development version is available directly from our [GitHub Repo](https://github.com/bmckernan/test_mcfacts). To start, clone the repository:

```bash
$ git clone https://github.com/bmckernan/test_mcfacts
$ cd test_mcfacts
```

#### Automated Setup

Contained in the `Makefile` are a few make commands to get everything setup and running.

> **Note:**
> If you are running on Windows or want to set things up manually, skip to the [manual](#manual-setup) section below.

```bash
# Create the conda environment and install required packages
$ make setup

# Activate the conda environment that was created for us
$ conda activate mcfacts-dev

# Run mcfacts_sim.py with default initial values, then run population_plots.py
$ make plots
```

Done! Below are some extra commands that you might find helpful

```bash
# Delete your runs directory and other output files
$ make clean

# Re install the mcfacts package for changes to take effect
$ make install
```

#### Manual Setup

We recommend that you create a Conda environment for working with McFACTS.
You can do this by running

```bash
# Create the conda environment with some default packages installed
$ conda create --name mcfacts-dev "python>=3.10.4" pip -c conda-forge -c defaults

# Activate the enviornment we just created
$ conda activate mcfacts-dev

# Install the mcfacts package
$ python -m pip install --editable .

# Now all that we have left to do is run McFACTS!
$ python mcfacts_sim.py --galaxy_num 10 --fname-ini ./recipes/model_choice_old.ini --fname-log out.log --seed 3456789012
```

Our default inputs are located at `./recipes/model_choice_old.ini`. Edit this file or create your own `model_choice.ini` file with different inputs.

To use a different ini file, replace the file path after the `--fname-ini` argument.

```bash
$ python mcfacts_sim.py --fname-ini /path/to/your/file.ini
```

### Output Files

Output files will appear in `runs`. For each timestep there will be an `output_bh_single_ts.dat` and `output_bh_binary_ts.dat` where `ts` is the timestep number (0-N)---these files track every single/binary in the simulation at that timestep. 

The entire run will have a single `output_mergers.dat` file, which gives the details of every merger throughout the run. If you are trying to get distributions of merger properties, you probably only need `output_mergers.dat`, but if you are trying to track the history of individual mergers or want to know e.g. the state of the nuclear star cluster after an AGN of some duration, you will want the larger output suite.

### Documentation
You can find documentation for our code and modules at our [Read the Docs](https://mcfacts.readthedocs.io).

Input and outputs are documented in `IOdocumentation.txt`. 

Want build or browse the docs locally? Run the following:

```bash
# Switch to the mcfacts-dev environment and install required packages to build the docs
$ conda activate mcfacts-dev
$ conda install sphinx sphinx_rtd_theme sphinx-autoapi sphinx-automodapi

# Switch to the docs directory
$ cd docs

# Clean up any previous generated docs 
$ make clean

# Generate the html version of the docs in ./docs/build/html
$ make html
```