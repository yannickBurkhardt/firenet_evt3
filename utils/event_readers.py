import pandas as pd
import zipfile
from os.path import splitext
import numpy as np
from .timers import Timer
from .evt3_decoder import decode_file, WORDS_PER_CHUNK
from .evt3_utils import get_evt_version_from_header

TXT_EXTENSIONS = ['.txt', '.zip']
RAW_EXTENSIONS = ['.raw']


def window_duration_to_microseconds(duration_ms):
    duration_us = int(round(duration_ms * 1000.0))
    assert duration_us > 0, 'The window duration must be at least 1 us'
    return duration_us


def absolute_window_index(timestamp_s, window_duration_us):
    """
    Index of the absolute time window a single event belongs to (see absolute_window_indices).
    """
    return int(round(timestamp_s * 1e6)) // window_duration_us


def absolute_window_indices(timestamps_s, window_duration_us):
    """
    Index of the absolute time window each event belongs to: window k holds the events whose
    timestamp falls in [k * duration, (k + 1) * duration).

    The grid is anchored at t = 0 and defined on integer microseconds, i.e. on the native
    resolution of the sensor, rather than on the floating point timestamps. Two recordings
    are therefore cut at exactly the same absolute times, no matter how their timestamps
    were rounded on the way to floating point.
    """
    return np.rint(timestamps_s * 1e6).astype(np.int64) // window_duration_us


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0, t_offset=0.0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.t_offset = t_offset
        self.iterator = pd.read_csv(path_to_event_file, sep=r'\s+', header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
        if self.t_offset != 0.0:
            event_window[:, 0] += self.t_offset
        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    Event windows are cut on the *absolute* time grid, i.e. window k contains the
    events whose timestamp falls in [k * duration, (k + 1) * duration).
    Two recordings processed with the same window duration are therefore cut at
    the same absolute times, which makes the resulting frames directly
    comparable across, e.g., the two cameras of a stereo rig.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0, t_offset=0.0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in TXT_EXTENSIONS)
        self.is_zip_file = (file_extension == '.zip')

        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], 'r')
        else:
            self.event_file = open(path_to_event_file, 'r')

        # ignore header + the first start_index lines
        for i in range(1 + start_index):
            self.event_file.readline()

        self.t_offset = t_offset
        self.duration_us = window_duration_to_microseconds(duration_ms)
        self.current_window_index = None
        self.event_list = []

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        with Timer('Reading event window from file'):
            for line in self.event_file:
                if self.is_zip_file:
                    line = line.decode("utf-8")
                t, x, y, pol = line.split(' ')
                t, x, y, pol = float(t) + self.t_offset, int(x), int(y), int(pol)
                window_index = absolute_window_index(t, self.duration_us)
                if self.current_window_index is None:
                    self.current_window_index = window_index
                if window_index > self.current_window_index:
                    self.current_window_index = window_index
                    event_window = np.array(self.event_list)
                    self.event_list = [[t, x, y, pol]]
                    return event_window
                self.event_list.append([t, x, y, pol])

            # end of file: flush the last (partial) window
            if self.event_list:
                event_window = np.array(self.event_list)
                self.event_list = []
                return event_window

        raise StopIteration


class Evt3EventBuffer:
    """
    Streams the events of an EVT3 '.raw' file, in chunks of decoded events.

    Events are returned as a [N x 4] float64 NumPy array, one event per row in
    the form [t, x, y, pol], with t in *seconds* (the absolute sensor timestamp,
    plus the optional t_offset; timestamps are never rebased to zero) and
    pol in {0, 1}.
    """

    def __init__(self, path_to_event_file, start_index=0, t_offset=0.0,
                 words_per_chunk=WORDS_PER_CHUNK):
        evt_version = get_evt_version_from_header(path_to_event_file)
        if evt_version is not None and not evt_version.startswith('3'):
            print('!!Warning!! {} declares event encoding version {}, but it will be decoded as EVT3.'.format(
                path_to_event_file, evt_version))

        self.path_to_event_file = path_to_event_file
        self.t_offset = t_offset
        self.num_events_to_skip = start_index
        self.num_events_decoded = 0
        self.chunk_iterator = decode_file(path_to_event_file, words_per_chunk=words_per_chunk)

    def next_chunk(self):
        """
        :return: the next chunk of events as a [N x 4] float64 array, or None at the end of the file.
        """
        for t, x, y, p in self.chunk_iterator:
            self.num_events_decoded += t.size
            if self.num_events_to_skip > 0:
                if self.num_events_to_skip >= t.size:
                    self.num_events_to_skip -= t.size
                    continue
                t, x, y, p = (a[self.num_events_to_skip:] for a in (t, x, y, p))
                self.num_events_to_skip = 0
            if t.size == 0:
                continue

            events = np.empty((t.size, 4), dtype=np.float64)
            # divide by 1e6 (exactly representable) rather than multiply by 1e-6 (not),
            # so that a microsecond timestamp maps to the same double as its decimal notation
            events[:, 0] = t.astype(np.float64) / 1e6 + self.t_offset
            events[:, 1] = x
            events[:, 2] = y
            events[:, 3] = p
            return events

        return None


class FixedSizeEvt3EventReader:
    """
    Reads events from a Prophesee EVT3 '.raw' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0, t_offset=0.0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.num_events = num_events
        self.buffer = Evt3EventBuffer(path_to_event_file, start_index=start_index, t_offset=t_offset)
        self.pending_events = None

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            while self.pending_events is None or len(self.pending_events) < self.num_events:
                chunk = self.buffer.next_chunk()
                if chunk is None:
                    break
                if self.pending_events is None:
                    self.pending_events = chunk
                else:
                    self.pending_events = np.concatenate((self.pending_events, chunk), axis=0)

            if self.pending_events is None or len(self.pending_events) == 0:
                raise StopIteration

            event_window = self.pending_events[:self.num_events]
            remaining_events = self.pending_events[self.num_events:]
            self.pending_events = remaining_events if len(remaining_events) > 0 else None

        return event_window


class FixedDurationEvt3EventReader:
    """
    Reads events from a Prophesee EVT3 '.raw' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    As for FixedDurationEventReader, windows are cut on the absolute time grid:
    window k contains the events whose timestamp falls in
    [k * duration, (k + 1) * duration). Windows that do not contain any event
    are skipped.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0, t_offset=0.0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        self.duration_us = window_duration_to_microseconds(duration_ms)
        self.buffer = Evt3EventBuffer(path_to_event_file, start_index=start_index, t_offset=t_offset)
        self.pending_events = None
        # index of the absolute time window each pending event belongs to; kept alongside
        # the events so that a window is always delimited by the very same criterion
        self.pending_window_indices = None
        self.reached_end_of_file = False

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            while True:
                if self.pending_events is not None and len(self.pending_events) > 0:
                    window_index = self.pending_window_indices[0]
                    # the events of a window are contiguous, since the file is time-ordered
                    num_events_in_window = int(np.searchsorted(self.pending_window_indices, window_index,
                                                              side='right'))
                    # the window is complete as soon as we have seen an event belonging to a
                    # later window, or if there is nothing left to read
                    if num_events_in_window < len(self.pending_events) or self.reached_end_of_file:
                        event_window = self.pending_events[:num_events_in_window]
                        self.pending_events = self.pending_events[num_events_in_window:]
                        self.pending_window_indices = self.pending_window_indices[num_events_in_window:]
                        return event_window

                if self.reached_end_of_file:
                    raise StopIteration

                chunk = self.buffer.next_chunk()
                if chunk is None:
                    self.reached_end_of_file = True
                    continue

                chunk_window_indices = absolute_window_indices(chunk[:, 0], self.duration_us)
                if self.pending_events is None or len(self.pending_events) == 0:
                    self.pending_events = chunk
                    self.pending_window_indices = chunk_window_indices
                else:
                    self.pending_events = np.concatenate((self.pending_events, chunk), axis=0)
                    self.pending_window_indices = np.concatenate((self.pending_window_indices,
                                                                  chunk_window_indices), axis=0)


def make_event_reader(path_to_event_file, fixed_duration, window_duration_ms=33.33, num_events=10000,
                      start_index=0, t_offset=0.0):
    """
    Instantiate the event reader matching the format of 'path_to_event_file'
    ('.raw' for Prophesee EVT3 files, '.txt' / '.zip' for text files).
    """
    file_extension = splitext(path_to_event_file)[1].lower()

    if file_extension in RAW_EXTENSIONS:
        reader_type = FixedDurationEvt3EventReader if fixed_duration else FixedSizeEvt3EventReader
    elif file_extension in TXT_EXTENSIONS:
        reader_type = FixedDurationEventReader if fixed_duration else FixedSizeEventReader
    else:
        raise ValueError('Unsupported event file extension: {} (expected one of: {})'.format(
            file_extension, ', '.join(RAW_EXTENSIONS + TXT_EXTENSIONS)))

    if fixed_duration:
        return reader_type(path_to_event_file, duration_ms=window_duration_ms,
                           start_index=start_index, t_offset=t_offset)

    return reader_type(path_to_event_file, num_events=num_events,
                       start_index=start_index, t_offset=t_offset)
