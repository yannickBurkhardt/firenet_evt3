import pandas as pd
import zipfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from os.path import splitext
import numpy as np
from .timers import Timer
from .evt3_decoder import decode_file, WORDS_PER_CHUNK
from .evt3_utils import get_evt_version_from_header

TXT_EXTENSIONS = ['.txt', '.zip']
RAW_EXTENSIONS = ['.raw']

# units a file of frame timestamps can be written in, in nanoseconds
FRAME_TIMESTAMP_UNITS_IN_NS = {'ns': 1, 'us': 1000, 's': 1000000000}


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


def read_frame_timestamps(path, unit='ns'):
    """
    Reads a file of frame timestamps, one per line, in the given unit: the times the frames are to
    be reconstructed at, e.g. the capture times of the images of another camera of the platform.

    They are kept as integer nanoseconds rather than as the seconds the rest of the pipeline works
    in, because an absolute timestamp is 19 digits long and no float holds that exactly, while the
    whole point of reconstructing at these times is that the frames land on them.

    :return: the timestamps as a strictly increasing int64 array [ns].
    """
    if unit not in FRAME_TIMESTAMP_UNITS_IN_NS:
        raise ValueError('Unknown timestamp unit: {} (expected one of: {})'.format(
            unit, ', '.join(sorted(FRAME_TIMESTAMP_UNITS_IN_NS))))
    factor = Decimal(FRAME_TIMESTAMP_UNITS_IN_NS[unit])

    timestamps = []
    with open(path, 'r') as timestamp_file:
        for line_number, line in enumerate(timestamp_file, start=1):
            line = line.split('#', 1)[0].strip().rstrip(',')
            if not line:
                continue
            try:
                timestamps.append(int((Decimal(line) * factor).to_integral_value(ROUND_HALF_EVEN)))
            except (InvalidOperation, ValueError):
                raise ValueError('{}, line {}: not a timestamp: {}'.format(path, line_number, line))

    if not timestamps:
        raise ValueError('{} holds no timestamp'.format(path))

    timestamps = np.array(timestamps, dtype=np.int64)
    # a timestamp that does not lie after the one before it would ask for a frame covering no
    # events at all, so it is dropped rather than reconstructed empty
    increasing = np.concatenate(([True], np.diff(timestamps) > 0))
    if not increasing.all():
        print('!!Warning!! {} of the {} timestamps of {} do not lie after the one before them, '
              'they are skipped.'.format((~increasing).sum(), timestamps.size, path))
        timestamps = timestamps[increasing]

    return timestamps


def event_timestamps_ns(timestamps_s):
    """
    The timestamps of decoded events as integer nanoseconds. Events are stamped in whole
    microseconds by the sensor, so the float64 seconds they are carried in are rounded back to
    that grid first (see absolute_window_indices).
    """
    return np.rint(timestamps_s * 1e6).astype(np.int64) * 1000


class FrameWindows:
    """
    The event windows a file of frame timestamps asks for: window k holds the events of
    (t[k-1], t[k]], so that the frame reconstructed from it is the state of the scene at t[k].

    The first window covers one frame period before the first timestamp, i.e. (t[0] - (t[1] -
    t[0]), t[0]]; the events before that are not part of any frame and are dropped. A window that
    holds no event at all yields no frame, as an empty window does with the other readers.
    """

    def __init__(self, timestamps_ns):
        self.timestamps_ns = timestamps_ns
        self.next_frame = 0
        self.num_empty_windows = 0
        # lower bound of the first window; without a second timestamp to give a frame period,
        # everything recorded before the first timestamp belongs to the first frame
        if timestamps_ns.size > 1:
            self.window_start_ns = int(2 * timestamps_ns[0] - timestamps_ns[1])
        else:
            self.window_start_ns = None

        print('Will cut the events at the {} timestamps of the frame timestamp file'.format(
            timestamps_ns.size))
        if timestamps_ns.size > 1:
            duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
            print('Output frame rate: {:.1f} Hz on average'.format(
                (timestamps_ns.size - 1) / duration_s if duration_s > 0 else float('inf')))

    def exhausted(self):
        return self.next_frame >= self.timestamps_ns.size

    def current_end_ns(self):
        return int(self.timestamps_ns[self.next_frame])

    def advance(self):
        """Moves on to the next frame, and returns the timestamp [ns] of the one just finished."""
        finished = self.current_end_ns()
        self.window_start_ns = finished
        self.next_frame += 1
        return finished

    def report_empty_windows(self):
        if self.num_empty_windows:
            print('!!Warning!! {} of the {} timestamps got no event at all, no frame was written '
                  'for them.'.format(self.num_empty_windows, self.timestamps_ns.size))


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


class TimestampedEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and cuts them at the timestamps of a frame
    timestamp file: one event window per timestamp, holding the events of (t[k-1], t[k]]
    (see FrameWindows).
    """

    def __init__(self, path_to_event_file, frame_timestamps_ns, start_index=0, t_offset=0.0):
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
        self.windows = FrameWindows(frame_timestamps_ns)
        self.frame_stamp_ns = None      # timestamp of the frame the last window is meant for
        self.frame_stamp = None         # the same timestamp in seconds
        self.event_list = []
        self.pending_event = None       # the event that ended the window, kept for the next one
        self.reached_end_of_file = False

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        with Timer('Reading event window from file'):
            while not self.windows.exhausted():
                end_ns = self.windows.current_end_ns()
                start_ns = self.windows.window_start_ns

                if self.pending_event is not None and self.pending_event[1] <= end_ns:
                    self.event_list.append(self.pending_event[0])
                    self.pending_event = None

                if self.pending_event is None and not self.reached_end_of_file:
                    self.reached_end_of_file = True
                    for line in self.event_file:
                        if self.is_zip_file:
                            line = line.decode("utf-8")
                        t, x, y, pol = line.split(' ')
                        t, x, y, pol = float(t) + self.t_offset, int(x), int(y), int(pol)
                        timestamp_ns = int(round(t * 1e6)) * 1000
                        if timestamp_ns > end_ns:  # the first event of a later window
                            self.pending_event = ([t, x, y, pol], timestamp_ns)
                            self.reached_end_of_file = False
                            break
                        if start_ns is None or timestamp_ns > start_ns:
                            self.event_list.append([t, x, y, pol])

                event_window = np.array(self.event_list) if self.event_list else None
                self.event_list = []
                self.frame_stamp_ns = self.windows.advance()
                self.frame_stamp = self.frame_stamp_ns / 1e9

                if event_window is None:
                    self.windows.num_empty_windows += 1
                    continue
                return event_window

        self.windows.report_empty_windows()
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


class TimestampedEvt3EventReader:
    """
    Reads events from a Prophesee EVT3 '.raw' file, and cuts them at the timestamps of a frame
    timestamp file: one event window per timestamp, holding the events of (t[k-1], t[k]]
    (see FrameWindows).

    This is what puts the reconstructed frames on the clock of another camera: with a '.raw' that
    was put on the time base of the platform, and the capture times of that camera as timestamps,
    every frame is the state of the scene at the instant the other camera took its image.
    """

    def __init__(self, path_to_event_file, frame_timestamps_ns, start_index=0, t_offset=0.0):
        self.windows = FrameWindows(frame_timestamps_ns)
        self.buffer = Evt3EventBuffer(path_to_event_file, start_index=start_index, t_offset=t_offset)
        self.frame_stamp_ns = None      # timestamp of the frame the last window is meant for
        self.frame_stamp = None         # the same timestamp in seconds
        self.pending_events = None
        # timestamps of the pending events, in whole nanoseconds; kept alongside the events so
        # that the windows are cut on integers rather than on the floats the events carry
        self.pending_ns = None
        self.reached_end_of_file = False

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            while not self.windows.exhausted():
                end_ns = self.windows.current_end_ns()

                # the window is only complete once an event beyond it has been read, or once
                # there is nothing left to read
                while not self.reached_end_of_file and (self.pending_ns is None
                                                        or self.pending_ns.size == 0
                                                        or self.pending_ns[-1] <= end_ns):
                    chunk = self.buffer.next_chunk()
                    if chunk is None:
                        self.reached_end_of_file = True
                        break
                    chunk_ns = event_timestamps_ns(chunk[:, 0])
                    if self.pending_events is None or self.pending_events.shape[0] == 0:
                        self.pending_events, self.pending_ns = chunk, chunk_ns
                    else:
                        self.pending_events = np.concatenate((self.pending_events, chunk), axis=0)
                        self.pending_ns = np.concatenate((self.pending_ns, chunk_ns), axis=0)

                event_window = None
                if self.pending_ns is not None and self.pending_ns.size > 0:
                    start_ns = self.windows.window_start_ns
                    first = 0 if start_ns is None else int(np.searchsorted(self.pending_ns,
                                                                           start_ns, side='right'))
                    last = int(np.searchsorted(self.pending_ns, end_ns, side='right'))
                    if last > first:
                        event_window = self.pending_events[first:last]
                    self.pending_events = self.pending_events[last:]
                    self.pending_ns = self.pending_ns[last:]

                self.frame_stamp_ns = self.windows.advance()
                self.frame_stamp = self.frame_stamp_ns / 1e9

                if event_window is None:
                    self.windows.num_empty_windows += 1
                    continue
                return event_window

        self.windows.report_empty_windows()
        raise StopIteration


def make_event_reader(path_to_event_file, fixed_duration, window_duration_ms=33.33, num_events=10000,
                      start_index=0, t_offset=0.0, frame_timestamps_ns=None):
    """
    Instantiate the event reader matching the format of 'path_to_event_file'
    ('.raw' for Prophesee EVT3 files, '.txt' / '.zip' for text files).

    'frame_timestamps_ns' cuts the events at the given frame timestamps (see read_frame_timestamps)
    instead of on a window grid, and takes precedence over 'fixed_duration'.
    """
    file_extension = splitext(path_to_event_file)[1].lower()

    if file_extension in RAW_EXTENSIONS:
        is_raw_file = True
    elif file_extension in TXT_EXTENSIONS:
        is_raw_file = False
    else:
        raise ValueError('Unsupported event file extension: {} (expected one of: {})'.format(
            file_extension, ', '.join(RAW_EXTENSIONS + TXT_EXTENSIONS)))

    if frame_timestamps_ns is not None:
        reader_type = TimestampedEvt3EventReader if is_raw_file else TimestampedEventReader
        return reader_type(path_to_event_file, frame_timestamps_ns,
                           start_index=start_index, t_offset=t_offset)

    if fixed_duration:
        reader_type = FixedDurationEvt3EventReader if is_raw_file else FixedDurationEventReader
        return reader_type(path_to_event_file, duration_ms=window_duration_ms,
                           start_index=start_index, t_offset=t_offset)

    reader_type = FixedSizeEvt3EventReader if is_raw_file else FixedSizeEventReader
    return reader_type(path_to_event_file, num_events=num_events,
                       start_index=start_index, t_offset=t_offset)
