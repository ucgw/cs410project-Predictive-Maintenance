## CS410 Project: Predictive Maintenance Hour Schedules by Summary


## Overview
This project attempts to offer suggested starting hours for proposed event summaries of maintenance tasks via term query strings using EM, LDA and LSA topic modeling. The raw dataset is somewhat contrived, however, loosely based off of actual scheduled maintenance days for a set of managed systems and services. The raw data has been cleaned for this project's use.
<br>
<br>
The utility for having self-serviced scheduling based on historical scheduled events can be derived by offering 2 suggested start hours to the user's query (that if we had more time, could provide feedback into the system for generating new topic models). A configuration file is included (`etc/run_model.ini`); used to change output behavior as well as support using different topic modeling algorithms and providing a generic visualization of topic densities and top terms in each topic.
<br>
<br>
The raw data consists of ics fields in json payloads. The fields apropos to this project are `SUMMARY` and `DTSTART`. They are used to generate the corpus of terms and their associated start hours across all of the raw data. A process takes all of the data and generates a metadata json payload that breaks down the tokens ("terms") and their associated start hours (Other data is included, however, it ended up being beyond the scope of what we had time to do).

## System Dependencies
* Python3   (>= Python 3.9)
* Bourne Shell   (bash)

## Python Dependencies
* numpy
* scipy
* matplotlib
* docopt
* configobj
* gensim

## `etc/run_model.ini` Configuration File
Below is an example configuration file that `run_model` uses from the repo
```
# configure which model to use
# valid methods: (em|lda|lsa)
[model]
  method = em
  debug = False
  show_viz = False
  datasource = './processed/emtopic-metads-202312071033.json'

[em_conf]
  viz_count = 4
  topic_count = 4
  save_model = False
```

## `run_model` Usage
```
Usage:
  run_model [--help] (--rand-query=<term_count>|--query=<query_string>)

Runs a model against the Dataset of Maintenance Event Schedules using one of the supported methods as defined in a configuration file:
  em:    generic EM (Expectation Maximization)
  lda:   LDA (Linear Discriminet Analysis)
  lsa:  LSA (Latent Semantic Analysis)

The purpose of the modeling is to offer as relevant Suggestions for the Best Hour of the day to Schedule an event based on query term strings evaluated.

Configuration File / Spec
=========================
# CFG_FILE Default: './etc/run_model.ini'
#
# To override to different ini config file path:

$ export _CFG_FILE=/path/to/run_model.ini


# CFG_SPEC Default: './share/run_model.spec'
#
# To override to different spec file path:

$ export _CFG_SPEC=/path/to/run_model.spec


Options:
  --rand-query=<term_count>    Number of random terms from corpus to generate a random query term string for generating hour suggestions
  --query=<query_string>       User defined term query string for generating hour suggestions
  --help        Print this help screen and exit.

NOTE: '--rand-query' and '--query' are mutually exclusive.
```

## `print_corpus` Usage
This command can be used to get a list of all the terms in the corpus printed to stdout (standard out); one term per newline.
<br>
<br>
Any combination of these terms can be used to craft a term query summary string passed into the `--query` parameter of `run_model`
```
$ ./print_corpus | head -5
scheduling
gpfs
license
system3
jupyter
```

## Instructions
1) Clone and Setup `venv` using python (>= 3.9)
```
$ git clone git@github.com:ucgw/cs410-project.git
$ cd cs410-project
$ python3 -m venv <desired venv directory path>
$ source <path to venv directory>/bin/activate
(venv) $ pip install -r requirements.txt
```

2) Generate metadata from the raw dataset in `./raw`<br>
NOTE: path will be displayed to STDOUT

```
$ ./gen_datasource
./processed/emtopic-metads-202312071033.json
```

3) Take the output from `gen_datasource` and add that as the value under `etc/run_model.ini` configuration section `[model]` configuration key `datasource`.

```
[model]
  ...
  datasource = './processed/emtopic-metads-202312071033.json'
```

### Running model queries
1) Determine what mode you want to run it in: `--query` or `--rand-query`

#### `--rand-query` Usage
```
# Randomly choose 5 terms from corpus for suggestions

$ ./run_model --rand-query=5
Query Tokens Processed: ['0287388', 'system8', '0145867', 'lustre', 'pbs']
Suggestions: [(('lustre', -3.844621964751844), (9, 85)), (('0287388', -1.7126241672093876), (11, 1))]
```

#### `--query` Usage
```
# User choosen terms from corpus for suggestions

./run_model --query="User1 reboot server after patch"
Query Tokens Processed: ['user1', 'reboot', 'server', 'after', 'patch']
Suggestions: [(('reboot', -5.518598398323515), (17, 25)), (('user1', -1.7126241672093876), (11, 1))]
```

### Interpreting `Suggestions` Output
The following information is embedded in the `Suggestions` list given back out. (Only 2 suggestions given)
```
First Decision (No Positional penalty; straight weighting)
Second Decision (Positional penalty; lower order => higher weight)

Example:
Decision 1. Term: 'reboot'. Hour: 17. Frequency: 25 
The term 'reboot' represents the top word in the query under the top topic in the distribution chosen
The value '17' represents Hour (in Military time) most representative of term
The value '25' represents the frequency of term
  
```

## Visualization Instructions
To see a visualization of topic breakdown (top k words per topic) as a plot, set the value under `etc/run_model.ini` configuration section `[model]` configuration key `show_viz` to `True`.
```
# configure which model to use
# valid methods: (em|lda)
[model]
  ...
  show_viz = True

```

Execute [run_model](#running-model-queries)

## Example Visualization:
![Topic Visualization](https://github.com/ucgw/cs410-project/blob/main/images/Figure_1.png?raw=true)
