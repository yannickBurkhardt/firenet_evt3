"""
Utilities to work with Prophesee EVT3 '.raw' files.

Only the ASCII header is parsed here; the binary event stream is decoded by
'utils/evt3_decoder.py' (see utils/event_readers.py).
"""

import re


def read_raw_header(path_to_event_file, max_header_bytes=8192):
    """
    Read the ASCII header of a '.raw' file.

    Each header line starts with '%'. The header ends at the first line that
    does not start with '%' (i.e. the beginning of the binary event stream).

    :return: a dict mapping lowercase header keys to their (string) values, plus
             the key '_lines' holding the raw header lines.
    """
    header = {'_lines': []}
    with open(path_to_event_file, 'rb') as f:
        raw_header = f.read(max_header_bytes)

    for raw_line in raw_header.split(b'\n'):
        if not raw_line.startswith(b'%'):
            break
        line = raw_line.decode('utf-8', errors='replace').lstrip('%').strip()
        header['_lines'].append(line)
        if not line:
            continue
        tokens = line.split(None, 1)
        key = tokens[0].strip().lower()
        value = tokens[1].strip() if len(tokens) > 1 else ''
        header[key] = value

    return header


def get_sensor_size_from_header(path_to_event_file):
    """
    Try to infer the sensor size from the header of a '.raw' file.

    Handles the header flavours found in the wild, e.g.
        % format EVT3;height=720;width=1280
        % geometry 1280x720
        % width 1280
        % height 720

    :return: (width, height), or None if the header does not specify a geometry.
    """
    header = read_raw_header(path_to_event_file)

    # '% format EVT3;height=720;width=1280'
    if 'format' in header:
        fields = dict()
        for field in header['format'].split(';'):
            if '=' in field:
                key, value = field.split('=', 1)
                fields[key.strip().lower()] = value.strip()
        if 'width' in fields and 'height' in fields:
            return int(fields['width']), int(fields['height'])

    # '% geometry 1280x720'
    if 'geometry' in header:
        match = re.match(r'^(\d+)\s*[xX]\s*(\d+)$', header['geometry'].strip())
        if match:
            return int(match.group(1)), int(match.group(2))

    # '% width 1280' + '% height 720'
    if 'width' in header and 'height' in header:
        return int(header['width']), int(header['height'])

    return None


def get_t_offset_from_header(path_to_event_file):
    """
    Read the time origin of a '.raw' file, i.e. the absolute time its timestamps are relative to.

    It is written by the synchronization tools as

        % t_offset_us 1782294822956474

    a Unix timestamp in microseconds, the same header key the 'metavision_evt2_raw_file_encoder'
    sample of OpenEB writes. Adding it to the timestamps of the recording puts them on the same
    (absolute) time base as the other sensors.

    :return: the time origin in seconds, or None if the header does not carry it.
    """
    header = read_raw_header(path_to_event_file)
    if 't_offset_us' not in header:
        return None

    value = header['t_offset_us'].split()[0] if header['t_offset_us'] else ''
    try:
        t_offset_us = int(value)
    except ValueError:
        print('!!Warning!! ignoring the malformed header line "% t_offset_us {}" of {}'.format(
            header['t_offset_us'], path_to_event_file))
        return None

    return t_offset_us / 1e6


def get_evt_version_from_header(path_to_event_file):
    """
    :return: the event encoding version declared in the header (e.g. '3.0'), or None.
    """
    header = read_raw_header(path_to_event_file)
    for key in ('evt', 'format'):
        if key in header:
            match = re.search(r'(?:EVT|evt)?\s*([0-9]+\.[0-9]+)', header[key])
            if match:
                return match.group(1)
    return None
