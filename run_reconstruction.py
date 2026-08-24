import torch
from utils.loading_utils import load_model, get_device
import argparse
from os.path import splitext
import pandas as pd
from utils.event_readers import make_event_reader, RAW_EXTENSIONS, absolute_window_index, \
    window_duration_to_microseconds
from utils.evt3_utils import get_sensor_size_from_header, get_t_offset_from_header
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--width', default=None, type=int,
                        help="Sensor width. Overrides the value read from the event file.")
    parser.add_argument('--height', default=None, type=int,
                        help="Sensor height. Overrides the value read from the event file.")
    parser.add_argument('--t_offset', default=None, type=float,
                        help="Offset added to every event timestamp, in seconds. Useful to bring the \
                              timestamps of several recordings (e.g. the two cameras of a stereo rig) \
                              into a common time frame. Defaults to the time origin declared in the \
                              header of a '.raw' file ('% t_offset_us'), and to 0 without one.")
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    path_to_events = args.input_file
    is_raw_file = splitext(path_to_events)[1].lower() in RAW_EXTENSIONS

    # Read the sensor size from the event file header ('.raw'), or from its first line ('.txt' / '.zip')
    if is_raw_file:
        sensor_size = get_sensor_size_from_header(path_to_events)
    else:
        header = pd.read_csv(path_to_events, sep=r'\s+', header=None, names=['width', 'height'],
                             dtype={'width': int, 'height': int},
                             nrows=1)
        sensor_size = tuple(header.values[0])

    if (args.width is None) != (args.height is None):
        raise ValueError('--width and --height must be specified together.')

    if args.width is not None:
        width, height = args.width, args.height
    elif sensor_size is not None:
        width, height = sensor_size
    else:
        raise ValueError('Could not read the sensor size from the header of {}. '
                         'Please specify it with --width and --height.'.format(path_to_events))
    print('Sensor size: {} x {}'.format(width, height))

    # Read the time origin of the recording from the event file header ('.raw'), unless the user
    # gave one explicitly. Synchronized recordings carry the absolute time of their first event, so
    # that the reconstructed frames end up stamped on the time base of the whole platform.
    t_offset = args.t_offset
    header_t_offset = get_t_offset_from_header(path_to_events) if is_raw_file else None
    if header_t_offset is not None:
        if t_offset is None:
            t_offset = header_t_offset
            print('Time origin read from the file header: {:.6f} s'.format(t_offset))
        elif t_offset != header_t_offset:
            print('!!Warning!! --t_offset {:.6f} s overrides the time origin of {:.6f} s declared '
                  'in the file header.'.format(t_offset, header_t_offset))
    if t_offset is None:
        t_offset = 0.0

    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    """ Read the events in windows """

    # Loop through the events and reconstruct images
    N = args.window_size
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    event_window_iterator = make_event_reader(path_to_events,
                                              fixed_duration=args.fixed_duration,
                                              window_duration_ms=args.window_duration,
                                              num_events=N,
                                              start_index=start_index,
                                              t_offset=t_offset)

    window_duration_us = window_duration_to_microseconds(args.window_duration) if args.fixed_duration else None

    with Timer('Processing entire dataset'):
        for event_window in event_window_iterator:

            last_timestamp = event_window[-1, 0]

            if args.fixed_duration:
                # Nominal timestamp of the frame: the end of the absolute time window its events
                # belong to. Unlike the timestamp of the last event, this is identical for two
                # recordings cut on the same grid, which is what makes the frames of a stereo pair
                # line up (e.g. when naming them with --image_name_format=timestamp_ns).
                window_index = absolute_window_index(event_window[0, 0], window_duration_us)
                frame_stamp = (window_index + 1) * window_duration_us / 1e6
            else:
                frame_stamp = last_timestamp

            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=model.num_bins,
                                                        width=width,
                                                        height=height)
                    event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)

            num_events_in_window = event_window.shape[0]
            reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp,
                                                frame_stamp=frame_stamp)

            start_index += num_events_in_window
