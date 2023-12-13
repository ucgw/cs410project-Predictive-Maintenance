#!/bin/sh

DS_TSFORMAT="%Y%m%d"
DS_TSDATE=`date +"${DS_TSFORMAT}"`
DS_TIMESTAMP=`date +"${DS_TSFORMAT}%H%M"`
DS_FILENAME="emtopic-metads-${DS_TIMESTAMP}.json"
DS_FULLPATH="./processed/${DS_FILENAME}"
DS_FILEMODE="0o0640"

python3<<! && echo $DS_FULLPATH

import json
import re
import os
import math
import numpy as np

from os import listdir, environ
from os.path import isfile, join, basename
from datetime import datetime, timedelta
from uuid import uuid4

class EmException(BaseException):
    pass

TOKEN_IGNORE = [
'and',
'for',
'not',
'are',
'can',
'till',
'non',
'over',
'from',
'the',
'when',
'that',
'only',
'all',
'out',
'ifications',
'some',
'quick',
'day',
'within',
'put',
'making',
'ker',
'cloudfl',
'aka',
'any',
'into',
'according',
'tor',
'mcp',
'rer',
'nel',
'need',
'tbd',
'remo',
'more',
'eir',
'rese',
'sent',
'inst',
'rine',
'encement',
'may',
'comes',
'exp',
'ify',
'bee',
'shd',
'ance',
'come',
'ocations',
'now',
'unt',
'breas',
'says',
'most',
'inea',
'well',
'rvation',
'with',
'acco',
'ing',
'unmo',
'str',
'add',
'ject1',
'umo',
'except',
'myxy',
'ths',
'manuy',
'moed',
'cess',
'till',
'but',
]

SCHEDULES_HISTDIR = './raw'
SCHEDULES_DTFORMAT = '%Y%m%dT%H%M%S'
SCHEDULES_DTSTART_KEYREGEX = re.compile(r'^(DTSTART.*)$')
SCHEDULES_DTEND_KEYREGEX = re.compile(r'^(DTEND.*)$')
SCHEDULES_REQUESTID_REGEX = re.compile(r'\#([0-9]+)')

TOKEN_IGNORE_REGEX = re.compile('|'.join(TOKEN_IGNORE), re.I)

def get_sched_files():
    return [
        join(SCHEDULES_HISTDIR, sched) \
          for sched in listdir(SCHEDULES_HISTDIR) \
            if isfile(join(SCHEDULES_HISTDIR, sched))
        ]

def get_dtstart_key(key_list):
    dtstart_key = None
    for k in key_list:
        key_check = SCHEDULES_DTSTART_KEYREGEX.search(k)
        if key_check:
            dtstart_key = key_check.group(1)
            break
    return dtstart_key

def get_request_id_from_event(event_summary):
    request_id = None
    requestid_check = SCHEDULES_REQUESTID_REGEX.search(event_summary)
    if requestid_check:
        request_id = requestid_check.group(1)
    else:
        request_id = str(uuid4())
    return request_id

def get_emtopic_tokens_from_event(event_summary, minlen=3):
    # first, strip out requestid (if it exists)
    #event_summary = SCHEDULES_REQUESTID_REGEX.sub('', event_summary)

    # next, remove PM_DOERS from the tokens to consider
    #event_summary = SCHEDULES_PM_DOERS_REGEX.sub('', event_summary)
    event_summary = TOKEN_IGNORE_REGEX.sub('', event_summary)

    # next, strip out any characters that are non-alphanumeric
    event_summary = re.sub(r'[^a-zA-Z0-9\s\-]|\-', ' ', event_summary)

    # next, strip out minlen - 1 non-space tokens from
    # consideration and normalize to single spaces between
    # tokens
    minlen_rx = re.compile(r'\b[^\s+]{,'+f'{minlen-1}'+r'}\b')
    event_summary = re.sub('\s+', ' ', minlen_rx.sub('', event_summary))

    event_summary = TOKEN_IGNORE_REGEX.sub('', event_summary)
    # next, strip out leading and trailing spaces from string
    # and lowercase everything
    event_summary = event_summary.lstrip()
    event_summary = event_summary.rstrip()
    event_summary = event_summary.lower()

    # finally, return the list of tokens from the summary
    return event_summary.split(' ')

def get_hours_operational(startdt, enddt):
    hour_span = 1
    hour = timedelta(hours=1)
    next_hourdt = startdt

    hour_ops = [startdt.hour]

    if enddt > startdt:
        time_delta = enddt - startdt
        time_delta_hours = time_delta.seconds/3600.0

        for i in range(int(time_delta_hours)):
            next_hourdt += hour
            hour_counter = next_hourdt.hour
            if enddt.hour != next_hourdt.hour:
                hour_ops.append(hour_counter)

        if startdt.hour != enddt.hour:
            if enddt.hour not in hour_ops:
                if enddt.minute > 0:
                    hour_ops.append(enddt.hour)

        #hour_ops[-1] = math.floor(np.mean(hour_ops))

    return list(hour_ops)

def get_dtend_key(key_list):
    dtend_key = None
    for k in key_list:
        key_check = SCHEDULES_DTEND_KEYREGEX.search(k)
        if key_check:
            dtend_key = key_check.group(1)
            break
    return dtend_key

def get_emtopic_metadata_from_event(event, dtstart_key, dtend_key):
    start_dp = None
    end_dp = None
    dur_dp = None
    hour_ops_dp = []
    event_meta = {}

    try:
        startdt = datetime.strptime(
                    event[dtstart_key],
                    SCHEDULES_DTFORMAT
                  )
        enddt = datetime.strptime(
                  event[dtend_key],
                  SCHEDULES_DTFORMAT
                )
        duration = enddt - startdt

        start_dp = int(startdt.strftime('%H%M'))
        end_dp = int(enddt.strftime('%H%M'))
        dur_dp = int(duration.total_seconds() / 60)
        hour_ops_dp = get_hours_operational(startdt, enddt)
    except KeyError:
        # try one last time to get a valid set of dt keys
        dtstart_key = get_dtstart_key(event.keys())
        dtend_key = get_dtend_key(event.keys())

        startdt = datetime.strptime(
                    event[dtstart_key],
                    SCHEDULES_DTFORMAT
                  )
        enddt = datetime.strptime(
                  event[dtend_key],
                  SCHEDULES_DTFORMAT
                )
        duration = enddt - startdt

        start_dp = int(startdt.strftime('%H%M'))
        end_dp = int(enddt.strftime('%H%M'))
        dur_dp = int(duration.total_seconds() / 60)
        hour_ops_dp = get_hours_operational(startdt, enddt)
    except Exception as err:
        raise EmException("BUG: inputs not processed as expected: exception: {err}".format(err=err))

    request_id = get_request_id_from_event(event['SUMMARY'])
    tokens = get_emtopic_tokens_from_event(event['SUMMARY'])
    tokens = [tok for tok in tokens if len(tok) > 0]

    event_meta['request_id'] = request_id
    event_meta['start_dp'] = start_dp
    event_meta['end_dp'] = end_dp
    event_meta['dur_dp'] = dur_dp
    event_meta['tokens'] = tokens
    event_meta['hour_ops'] = hour_ops_dp

    return event_meta
    
def get_data_from_json(sched_file):
    sched_data = []
    sched_events = {}

    try:
        with open(sched_file) as scfh:
            sched_events = json.load(scfh)
    except (IOError, json.decoder.JSONDecodeError) as err:
        raise EmException("BUG: {f} couldn't be processed for json content".format(f=sched_file))

    example_record = sched_events[0].keys()

    dtstart_key = get_dtstart_key(example_record)
    dtend_key = get_dtend_key(example_record)

    for sched_event in sched_events:
        event_data = get_emtopic_metadata_from_event(
                       sched_event,
                       dtstart_key,
                       dtend_key
                     )

        if len(event_data) > 0:
            sched_data.append(event_data)

    return sched_data

def get_training_data():
    sched_files = get_sched_files()
    data_array = []

    for sched in sched_files:
        data_array.append(get_data_from_json(sched))

    return data_array


if __name__ == '__main__':
    with open('$DS_FULLPATH', 'w+') as dsfh:
        json.dump(get_training_data(), dsfh)
    os.chmod('$DS_FULLPATH', int($DS_FILEMODE))
    #training_data = get_training_data()
    #print(training_data)

!
