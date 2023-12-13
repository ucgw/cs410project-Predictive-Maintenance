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
exists
elastic
ntp
states
attrs
```

## Instructions
1) Clone and Setup `venv` using python (>= 3.9)
```
$ git clone git@github.com:ucgw/cs410project-Predictive-Maintenance
$ cd cs410project-Predictive-Maintenance
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
# Randomly choose 5 terms from corpus for suggestions using EM

$ ./run_model --rand-query=5
Query Tokens Processed: ['warn', '0143798', 'work', 'system14', 'system1']
Topic Model Method: EM
Suggestions:
Decision 1. Term: 'system1'. Hour: 9. Frequency: 53 
Decision 2. Term: '0143798'. Hour: 21. Frequency: 12 
Top words Topic[2]: ['system3', 'release', 'test', 'users']


# Randomly choose 7 terms from the corpus for suggestions using LDA

./run_model --rand-query=7
Query Tokens Processed: ['0152789', '0244125', 'vendor', 'mainten', 'system74', 'probable', 'images']
Topic Model Method: LDA
Suggestions:
Decision 1. Term: '0152789'. Hour: 21. Frequency: 10 
Decision 2. Term: 'images'. Hour: 14. Frequency: 8 
Top words Topic[0]: ['service27', 'environment', 'client', 'switch']
```

#### `--query` Usage
```
# User choosen terms from corpus for suggestions using EM

./run_model --query="User1 reboot server after patch"
Query Tokens Processed: ['user1', 'reboot', 'server', 'after', 'patch']
Topic Model Method: EM
Suggestions:
Decision 1. Term: 'reboot'. Hour: 17. Frequency: 25 
Decision 2. Term: 'user1'. Hour: 12. Frequency: 1 
Top words Topic[2]: ['system3', 'release', 'test', 'users']


# User choosen terms from corpus for suggestions using LDA

./run_model --query="User1 reboot server after patch"
Query Tokens Processed: ['user1', 'reboot', 'server', 'after', 'patch']
Topic Model Method: LDA
Suggestions:
Decision 1. Term: 'server'. Hour: 11. Frequency: 11
Decision 2. Term: 'patch'. Hour: 10. Frequency: 24
Top words Topic[2]: ['switch', 'client', '0200489', 'exclusion']
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
# valid methods: (em|lda|lsa)
[model]
  ...
  show_viz = True

```

Execute [run_model](#running-model-queries)

## Example Visualization:
![Topic Visualization](https://github.com/ucgw/cs410project-Predictive-Maintenance/blob/main/images/Figure_1.png?raw=true)
