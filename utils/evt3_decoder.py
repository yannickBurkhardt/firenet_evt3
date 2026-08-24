"""
Vectorized decoder for the CD events of a Prophesee EVT3 '.raw' file.

Why not use a library: 'expelliarmus' 1.1.12 decodes the timestamps of the EVK4 (IMX636) files
of this project incorrectly -- it doubles the contribution of the EVT_TIME_HIGH words, so a 52 s
recording comes out as 104 s, while the events themselves (count, x, y, polarity) are right.
This decoder follows the reference implementation of OpenEB
('standalone_samples/metavision_evt3_raw_file_decoder'), and is checked against
'metavision_file_to_csv' in tests/test_evt3_decoder.py.

The decoding is done with NumPy, one buffer of 16-bit words at a time, without any Python loop
over the events.
"""

import numpy as np

# EVT3 word types (the 4 most significant bits of each 16-bit word)
EVT_ADDR_Y = 0x0
EVT_ADDR_X = 0x2
VECT_BASE_X = 0x3
VECT_12 = 0x4
VECT_8 = 0x5
EVT_TIME_LOW = 0x6
EVT_TIME_HIGH = 0x8
EXT_TRIGGER = 0xA

PAYLOAD_MASK = 0x0FFF
COORD_MASK = 0x07FF
POLARITY_SHIFT = 11
TIME_HIGH_SHIFT = 12

# The 12 bits of an EVT_TIME_HIGH word are the bits 23..12 of the timestamp, so the time base wraps
# around every TIME_LOOP us and has to be unwrapped. Of the two readings of a time base that went
# down -- it wrapped around, or it stepped back -- the one with the smaller step is taken, so a wrap
# is recognized when the time base drops by more than half of the loop.
#
# The OpenEB standalone sample only recognizes a wrap when the time base drops by nearly a whole
# loop (4085 of the 4096 steps), to stay safe against a transmission error. That misses the wrap of
# a recording that has no event for a while across it: a stream stepping 4090 -> 16, i.e. an 86 ms
# gap over the wrap, then decodes 16.7 s in the past and time runs backwards from there. The
# Metavision SDK does not have that problem, and neither does this rule.
MAX_TIMESTAMP_BASE = ((1 << 12) - 1) << TIME_HIGH_SHIFT
TIME_LOOP = MAX_TIMESTAMP_BASE + (1 << TIME_HIGH_SHIFT)
MIN_LOOP_DROP = (1 << 12) // 2

VECT_12_BITS = 12
VECT_8_BITS = 8

WORDS_PER_CHUNK = 1 << 22  # 4M words = 8 MB per read
MAX_HEADER_BYTES = 1 << 20


def _bit_tables():
    """
    :return: (positions, counts), where positions[mask, k] is the position of the k-th least
             significant bit set in mask, and counts[mask] the number of bits set in it.
    """
    masks = np.arange(1 << VECT_12_BITS, dtype=np.int64)
    bits = (masks[:, None] >> np.arange(VECT_12_BITS)) & 1
    # sorting 'not set' puts the positions of the bits that are set first, in increasing order
    positions = np.argsort(bits == 0, axis=1, kind='stable').astype(np.uint8)
    return positions, bits.sum(axis=1).astype(np.int32)


BIT_POSITIONS, BIT_COUNTS = _bit_tables()


def find_header_end(path_to_event_file):
    """
    :return: the offset of the first byte of the binary event stream, i.e. the length of the
             ASCII header (each header line starts with '%').
    """
    offset = 0
    with open(path_to_event_file, 'rb') as f:
        while offset < MAX_HEADER_BYTES:
            line = f.readline()
            if not line or not line.startswith(b'%'):
                break
            offset += len(line)
    return offset


def _before(up_to_here, vector_positions, positions):
    """
    :param up_to_here: cumulative count over the vector words, up to and including each of them.
    :return: the count reached right before each of 'positions'.
    """
    index = np.searchsorted(vector_positions, positions)
    return np.where(index > 0, up_to_here[np.clip(index - 1, 0, None)], 0)


def _state_of(positions, values, queries, fallback):
    """
    Value of the state-setting word that applies to each query position.

    :param positions: sorted positions of the words that set the state, in this buffer.
    :param values: value each of them sets the state to.
    :param queries: sorted positions to look the state up at.
    :param fallback: value carried over from the previous buffer.
    :return: (value, position) arrays; the position is -1 where the fallback was used.
    """
    ordinal = np.searchsorted(positions, queries, side='right') - 1
    known = ordinal >= 0
    clipped = np.clip(ordinal, 0, None)
    if positions.size == 0:
        return np.full(queries.size, fallback, dtype=np.int64), np.full(queries.size, -1)
    return np.where(known, values[clipped], fallback), np.where(known, positions[clipped], -1)


class Evt3Decoder:
    """
    Decodes the CD events of an EVT3 stream, buffer by buffer. The state needed to carry the
    decoding over from one buffer to the next (time base, y, vector base) is kept here, so the
    buffers can be cut anywhere.

    Like the reference decoder, the events that precede the first EVT_TIME_HIGH word of the
    stream are dropped, since their timestamp is unknown, and the EXT_TRIGGER words are ignored.
    """

    def __init__(self):
        self.time_base = 0             # us contributed by the last EVT_TIME_HIGH word
        self.time_high_value = 0       # payload of the last EVT_TIME_HIGH word (12 bits)
        self.num_time_loops = 0        # number of times the time base wrapped around
        self.time_base_set = False     # whether an EVT_TIME_HIGH word was seen at all
        self.time_low = 0              # payload of the last EVT_TIME_LOW word (12 bits)
        self.time_low_valid = False    # whether that word came after the last EVT_TIME_HIGH
        self.y = 0                     # payload of the last EVT_ADDR_Y word
        self.base_x = 0                # x of the next bit of a vector event
        self.polarity = 0              # polarity of the current vector event

    def decode(self, words):
        """
        :param words: uint16 array of EVT3 words.
        :return: (t, x, y, p) arrays of the CD events of this buffer, t in us.
        """
        empty = (np.empty(0, dtype=np.int64),) * 4
        if words.size == 0:
            return empty

        word_types = (words >> 12).astype(np.uint8)
        payloads = words & PAYLOAD_MASK
        is_addr_x = word_types == EVT_ADDR_X
        is_vect_12 = word_types == VECT_12
        is_vect_8 = word_types == VECT_8

        # The words that only set the decoder state are few, so the state that applies to an event
        # is looked up in these compact arrays rather than expanded over the whole buffer.
        time_high_positions = np.flatnonzero(word_types == EVT_TIME_HIGH)
        time_low_positions = np.flatnonzero(word_types == EVT_TIME_LOW)
        addr_y_positions = np.flatnonzero(word_types == EVT_ADDR_Y)
        base_x_positions = np.flatnonzero(word_types == VECT_BASE_X)
        vector_positions = np.flatnonzero(is_vect_12 | is_vect_8)

        # --- time bases, one per EVT_TIME_HIGH word --------------------------------------
        time_high_values = payloads[time_high_positions].astype(np.int64)
        if time_high_values.size:
            previous = np.concatenate(([self.time_high_value], time_high_values[:-1]))
            loops = self.num_time_loops + np.cumsum(previous - time_high_values >= MIN_LOOP_DROP)
            time_bases = (time_high_values << TIME_HIGH_SHIFT) + loops * TIME_LOOP
        else:
            loops = np.empty(0, dtype=np.int64)
            time_bases = np.empty(0, dtype=np.int64)
        time_low_values = payloads[time_low_positions].astype(np.int64)

        # --- how many events each word produces -----------------------------------------
        vector_masks = np.where(is_vect_12[vector_positions], payloads[vector_positions],
                                payloads[vector_positions] & 0xFF).astype(np.int64)
        num_events_per_word = is_addr_x.astype(np.int32)
        num_events_per_word[vector_positions] = BIT_COUNTS[vector_masks]
        if not self.time_base_set:
            # no time base yet: drop the events until the first EVT_TIME_HIGH word
            num_events_per_word[:time_high_positions[0] if time_high_values.size else words.size] = 0

        num_events = int(num_events_per_word.sum())
        if num_events == 0:
            self._carry_state(words, is_vect_12, is_vect_8, payloads, time_high_positions,
                              time_high_values, time_bases, loops, time_low_positions,
                              time_low_values, addr_y_positions, base_x_positions,
                              vector_positions)
            return empty

        source = np.repeat(np.arange(words.size, dtype=np.int32), num_events_per_word)
        starts = np.cumsum(num_events_per_word) - num_events_per_word
        rank = np.arange(num_events, dtype=np.int32) - starts[source]

        # --- timestamps -------------------------------------------------------------------
        base, base_position = _state_of(time_high_positions, time_bases, source, self.time_base)
        low, low_position = _state_of(time_low_positions, time_low_values, source, self.time_low)
        # an EVT_TIME_HIGH word resets the timestamp to the time base: the low bits only count
        # from the next EVT_TIME_LOW word on
        low_is_current = np.where((low_position < 0) & (base_position < 0),
                                  self.time_low_valid, low_position > base_position)
        t = base + np.where(low_is_current, low, 0)

        # --- coordinates and polarities ---------------------------------------------------
        y, _ = _state_of(addr_y_positions, payloads[addr_y_positions] & COORD_MASK, source, self.y)

        single = is_addr_x[source]
        x = (payloads[source] & COORD_MASK).astype(np.int64)
        p = (payloads[source] >> POLARITY_SHIFT).astype(np.int64)
        if vector_positions.size:
            base_x, polarity = self._vector_bases(is_vect_12, is_vect_8, payloads,
                                                  base_x_positions, vector_positions)
            index = np.searchsorted(vector_positions, source)
            np.clip(index, 0, vector_positions.size - 1, out=index)
            x = np.where(single, x, base_x[index] + BIT_POSITIONS[vector_masks[index], rank])
            p = np.where(single, p, polarity[index])

        self._carry_state(words, is_vect_12, is_vect_8, payloads, time_high_positions,
                          time_high_values, time_bases, loops, time_low_positions,
                          time_low_values, addr_y_positions, base_x_positions, vector_positions)
        return t, x, y, p

    def _vector_bases(self, is_vect_12, is_vect_8, payloads, base_x_positions,
                      vector_positions):
        """
        Every VECT_12 / VECT_8 word moves the vector base by 12 / 8 pixels, so the base of a
        vector word is the base of the last VECT_BASE_X word plus the vector words in between.

        :return: (base_x, polarity) of each vector word of the buffer.
        """
        base_x, _ = _state_of(base_x_positions, payloads[base_x_positions] & COORD_MASK,
                              vector_positions, self.base_x)
        polarity, _ = _state_of(base_x_positions, payloads[base_x_positions] >> POLARITY_SHIFT,
                                vector_positions, self.polarity)

        moved = np.zeros(vector_positions.size, dtype=np.int64)
        for flags, width in ((is_vect_12, VECT_12_BITS), (is_vect_8, VECT_8_BITS)):
            is_kind = flags[vector_positions]
            up_to_here = np.cumsum(is_kind, dtype=np.int64)
            before_here = up_to_here - is_kind
            # how many words of this kind precede the VECT_BASE_X word each vector word belongs to
            at_base, _ = _state_of(base_x_positions,
                                   _before(up_to_here, vector_positions, base_x_positions),
                                   vector_positions, 0)
            moved += width * (before_here - at_base)

        return base_x + moved, polarity

    def _carry_state(self, words, is_vect_12, is_vect_8, payloads, time_high_positions,
                     time_high_values, time_bases, loops, time_low_positions, time_low_values,
                     addr_y_positions, base_x_positions, vector_positions):
        """Remembers what the next buffer needs to continue the decoding."""
        if time_high_values.size:
            self.time_base = int(time_bases[-1])
            self.time_high_value = int(time_high_values[-1])
            self.num_time_loops = int(loops[-1])
            self.time_base_set = True
        if time_low_values.size:
            self.time_low = int(time_low_values[-1])
        last_time_high = time_high_positions[-1] if time_high_positions.size else -1
        last_time_low = time_low_positions[-1] if time_low_positions.size else -1
        if last_time_high >= 0 or last_time_low >= 0:
            self.time_low_valid = bool(last_time_low > last_time_high)
        if addr_y_positions.size:
            self.y = int(payloads[addr_y_positions[-1]] & COORD_MASK)
        if base_x_positions.size:
            self.polarity = int(payloads[base_x_positions[-1]] >> POLARITY_SHIFT)

        # base of a vector word that would follow this buffer
        num_12 = int(is_vect_12[vector_positions].sum()) if vector_positions.size else 0
        num_8 = vector_positions.size - num_12
        if base_x_positions.size:
            after = vector_positions > base_x_positions[-1]
            num_12_after = int(is_vect_12[vector_positions[after]].sum())
            num_8_after = int(after.sum()) - num_12_after
            self.base_x = int(payloads[base_x_positions[-1]] & COORD_MASK) + \
                VECT_12_BITS * num_12_after + VECT_8_BITS * num_8_after
        else:
            self.base_x += VECT_12_BITS * num_12 + VECT_8_BITS * num_8


def decode_file(path_to_event_file, words_per_chunk=WORDS_PER_CHUNK):
    """
    Decode the CD events of an EVT3 '.raw' file, chunk by chunk.

    The timestamps are checked to never go back: a stream that is not decoded correctly is much
    easier to recognize here than downstream, where events of the whole recording pile up in a
    single time window until something runs out of memory.

    :return: an iterator over the (t, x, y, p) int64 arrays of the file, t in us.
    """
    decoder = Evt3Decoder()
    last_timestamp = None
    with open(path_to_event_file, 'rb') as f:
        f.seek(find_header_end(path_to_event_file))
        while True:
            words = np.fromfile(f, dtype='<u2', count=words_per_chunk)
            if words.size == 0:
                return
            t, x, y, p = decoder.decode(words)
            if t.size:
                backwards = np.flatnonzero(np.diff(t) < 0)
                if last_timestamp is not None and t[0] < last_timestamp:
                    raise RuntimeError(
                        '{}: the timestamps go back from {} us to {} us, the stream is not '
                        'decoded correctly.'.format(path_to_event_file, last_timestamp, t[0]))
                if backwards.size:
                    index = backwards[0]
                    raise RuntimeError(
                        '{}: the timestamps go back from {} us to {} us, the stream is not '
                        'decoded correctly.'.format(path_to_event_file, t[index], t[index + 1]))
                last_timestamp = int(t[-1])
            yield t, x, y, p
