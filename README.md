# High Speed and High Dynamic Range Video with an Event Camera

[![High Speed and High Dynamic Range Video with an Event Camera](http://rpg.ifi.uzh.ch/E2VID/video_thumbnail.png)](https://youtu.be/eomALySSGVU)

This is the code for the paper **High Speed and High Dynamic Range Video with an Event Camera** by [Henri Rebecq](http://henri.rebecq.fr), Rene Ranftl, [Vladlen Koltun](http://vladlen.info/) and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html):

You can find a pdf of the paper [here](http://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf).

It also includes the [**FireNet**](https://www.cedricscheerlinck.com/firenet) model (from the
`cedric/firenet` branch of this fork), a much lighter and faster variant of E2VID.

If you use any of this code, please cite the following publications:

```bibtex
@InProceedings{Scheerlinck20wacv,
  author        = {Cedric Scheerlinck and Henri Rebecq and Daniel Gehrig and Nick Barnes and Robert Mahony and Davide Scaramuzza},
  title         = {Fast Image Reconstruction with an Event Camera},
  booktitle     = {{IEEE} Winter Conf. Appl. Comput. Vis. {(WACV)}},
  year          = {2020},
  pages         = {156--163}
}
```

```bibtex
@Article{Rebecq19pami,
  author        = {Henri Rebecq and Ren{\'{e}} Ranftl and Vladlen Koltun and Davide Scaramuzza},
  title         = {High Speed and High Dynamic Range Video with an Event Camera},
  journal       = {{IEEE} Trans. Pattern Anal. Mach. Intell. (T-PAMI)},
  url           = {http://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf},
  year          = 2019
}
```


```bibtex
@Article{Rebecq19cvpr,
  author        = {Henri Rebecq and Ren{\'{e}} Ranftl and Vladlen Koltun and Davide Scaramuzza},
  title         = {Events-to-Video: Bringing Modern Computer Vision to Event Cameras},
  journal       = {{IEEE} Conf. Comput. Vis. Pattern Recog. (CVPR)},
  year          = 2019
}
```

## Install

Dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.0
- [NumPy](https://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [OpenCV](https://opencv.org/)

### Install with Anaconda

The installation requires [Anaconda3](https://www.anaconda.com/distribution/).
You can create a new Anaconda environment with the required dependencies as follows (make sure to adapt the CUDA version according to your setup):

```bash
conda create -n e2vid python=3.10
conda activate e2vid
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas opencv-python scipy
```

## Run

- Download a pretrained model. Either the E2VID model:

```bash
wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O pretrained/E2VID_lightweight.pth.tar
```

or the [FireNet model](https://drive.google.com/file/d/1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa/view?usp=sharing)
(`firenet_1000.pth.tar`, to be placed in `pretrained/`). Both are loaded with `-c` and need no
further flags: the architecture is read from the checkpoint.

- Download an example file with event data:

```bash
wget "http://rpg.ifi.uzh.ch/data/E2VID/datasets/ECD_IJRR17/dynamic_6dof.zip" -O data/dynamic_6dof.zip
```

Before running the reconstruction, make sure the conda environment is sourced:

```bash
conda activate e2vid
```

- Run reconstruction:

```bash
python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i data/dynamic_6dof.zip \
  --auto_hdr \
  --display \
  --show_events
```

### Event data formats

`--input_file` / `-i` accepts:

- **Prophesee EVT3 `.raw` files** (e.g. recordings from an EVK / IMX636 sensor), decoded by
  `utils/evt3_decoder.py`, a NumPy port of the reference decoder of
  [OpenEB](https://github.com/prophesee-ai/openeb) (`metavision_evt3_raw_file_decoder`). The file is
  decoded in a streaming fashion, so recordings of arbitrary length can be processed with a constant
  memory footprint. EXT_TRIGGER words are ignored, so recordings with external trigger signals can be
  read as they are. (The decoder replaces `expelliarmus`, which doubles the contribution of the
  EVT_TIME_HIGH words on these files: a 52 s recording decodes as 104 s.)
  The sensor size is read from the `.raw` header (`% format EVT3;height=720;width=1280` or
  `% geometry 1280x720`); if the header does not specify it, pass `--width` and `--height`.
- **Text files** (`.txt`, or a `.zip` archive containing a single `.txt` file), in the format used by the
  [Event Camera Dataset](http://rpg.ifi.uzh.ch/davis_data.html): a first line with the sensor size
  (`width height`), followed by one event per line as `timestamp x y polarity`, with the timestamp in seconds.

```bash
python run_reconstruction.py \
  -c pretrained/E2VID_lightweight.pth.tar \
  -i data/my_recording.raw \
  --fixed_duration -T 33.33 \
  --auto_hdr
```

Event timestamps are never rebased: the timestamps written to `timestamps.txt` are the timestamps of
the recording itself (seconds; for `.raw` files, the sensor timestamps converted from microseconds),
shifted by `--t_offset`. Together with the fact that `--fixed_duration` windows are cut on
an absolute time grid (see `--window_duration` below), this means two recordings reconstructed with
the same `-T` are frame-aligned, e.g. for the two cameras of a stereo rig.

A `.raw` file whose header declares a time origin,

```
% t_offset_us 1782294822956474
```

is shifted by it automatically, so that its frames are stamped with absolute (Unix) time. That is
the header key written by the `metavision_evt2_raw_file_encoder` sample of OpenEB, and by the
synchronization tools that put a recording on the time base of an external clock. Pass `--t_offset`
explicitly to override it.

## Parameters

Below is a description of the most important parameters:

#### Main parameters

- ``--window_size`` / ``-N`` (default: None) Number of events per window. This is the parameter that has the most influence of the image reconstruction quality. If set to None, this number will be automatically computed based on the sensor size, as N = width * height * num_events_per_pixel (see description of that parameter below). Ignored if `--fixed_duration` is set.
- ``--fixed_duration`` (default: False) If True, will use windows of events with a fixed duration (i.e. a fixed output frame rate).
- ``--window_duration`` / ``-T`` (default: 33 ms) Duration of each event window, in milliseconds. The value of this parameter has strong influence on the image reconstruction quality. Its value may need to be adapted to the dynamics of the scene. Ignored if `--fixed_duration` is not set. Windows are cut on the absolute time grid anchored at t = 0, i.e. window k contains the events whose timestamp falls in [k * T, (k + 1) * T), so that the window boundaries do not depend on when the recording happens to start. Windows without any event produce no output frame.
- ``--width``, ``--height`` (default: None) Sensor size. Overrides the value read from the event file, and required for `.raw` files whose header does not specify a geometry. Must be given together.
- ``--t_offset`` (default: the time origin declared in the header of a `.raw` file, 0.0 without one) Offset in seconds added to every event timestamp, to bring several recordings into a common time frame (e.g. to compensate a known delay between the two cameras of a stereo rig, or to stamp them with absolute time). It shifts both the output timestamps and the window boundaries. Given explicitly, it replaces the `% t_offset_us` value of the header rather than adding to it.
- ``--Imin`` (default: 0.0), `--Imax` (default: 1.0): linear tone mapping is performed by normalizing the output image as follows: `I = (I - Imin) / (Imax - Imin)`. If `--auto_hdr` is set to True, `--Imin` and `--Imax` will be automatically computed as the min (resp. max) intensity values.
- ``--auto_hdr`` (default: False) Automatically compute `--Imin` and `--Imax`. Disabled when `--color` is set.
- ``--color`` (default: False): if True, will perform color reconstruction as described in the paper. Only use this with a [color event camera](http://rpg.ifi.uzh.ch/CED.html) such as the Color DAVIS346.

#### Output parameters

- ``--output_folder``: path of the output folder. The reconstructed images are written **directly** into it (no sub-folder), together with `timestamps.txt`. If not set, the image reconstructions will not be saved to disk.
- ``--image_name_format`` (default: 'timestamp_ns'): how to name the reconstructed images. `timestamp_ns` gives `<timestamp in ns>.png`, zero padded to 19 digits, which is the [image folder format expected by Kalibr](https://github.com/ethz-asl/kalibr/wiki/bag-format); `index` gives `frame_0000000000.png`. See below.

#### Kalibr image folder format

With `--image_name_format timestamp_ns` (the default), the images are named after their timestamp in
nanoseconds, so that `kalibr_bagcreater` can read them directly. Point `--output_folder` at the camera
folder of the dataset directory:

```bash
for cam in cam0 cam1; do
  python run_reconstruction.py -c pretrained/firenet_1000.pth.tar \
    -i /path/to/${cam}.raw --fixed_duration -T 50.0 --auto_hdr \
    -o /path/to/dataset-dir/${cam}
done
kalibr_bagcreater --folder /path/to/dataset-dir --output-bag calib.bag
```

Notes:

- The name of a frame is the **end of its absolute time window** (`(k + 1) * T`), not the timestamp of
  its last event. Two cameras reconstructed with the same `-T` therefore produce exactly the same file
  names, which is what makes the stereo pair line up. `timestamps.txt` keeps reporting the timestamp of
  the last event of each window.
- Kalibr parses a file name as `secs = name[:-9]`, `nsecs = name[-9:]`, so names shorter than 10 digits
  break it; the 19 digit zero padding avoids this for recordings whose timestamps start near zero.
- `kalibr_bagcreater` walks the camera folder recursively and picks up every image it finds. The only
  other file written there is `timestamps.txt`, which it ignores (it only collects `.bmp` / `.png` /
  `.jpg`). Event previews are never written to disk.
- Recordings synchronized against an external clock carry their time origin in the header
  (`% t_offset_us`) and are stamped with absolute time without any extra flag. Otherwise, use
  `--t_offset` to move the recordings of the two cameras into a common time frame; a negative
  resulting timestamp is refused, since it cannot be expressed as a file name.

#### Display parameters

- ``--display`` (default: False): display the video reconstruction in real-time in an OpenCV window.
- ``--show_events`` (default: False): show the input events side-by-side with the reconstruction. The event previews are only displayed, never written to disk.

#### Additional parameters

- ``--num_events_per_pixel`` (default: 0.35): Parameter used to automatically estimate the window size based on the sensor size. The value of 0.35 was chosen to correspond to ~ 15,000 events on a 240x180 sensor such as the DAVIS240C.
- ``--no-normalize`` (default: False): Disable event tensor normalization: this will improve speed a bit, but might degrade the image quality a bit.
- ``--no-recurrent`` (default: False): Disable the recurrent connection (i.e. do not maintain a state). For experimenting only, the results will be flickering a lot.
- ``--hot_pixels_file`` (default: None): Path to a file specifying the locations of hot pixels (such a file can be obtained with [this tool](https://github.com/cedric-scheerlinck/dvs_tools/tree/master/dvs_hot_pixel_filter) for example). These pixels will be ignored (i.e. zeroed out in the event tensors).

## Example datasets

We provide a list of example (publicly available) event datasets to get started with E2VID.

- [High Speed (gun shooting!) and HDR Dataset](http://rpg.ifi.uzh.ch/E2VID.html)
- [Event Camera Dataset](http://rpg.ifi.uzh.ch/data/E2VID/datasets/ECD_IJRR17/)
- [Bardow et al., CVPR'16](http://rpg.ifi.uzh.ch/data/E2VID/datasets/SOFIE_CVPR16/)
- [Scherlinck et al., ACCV'18](http://rpg.ifi.uzh.ch/data/E2VID/datasets/HF_ACCV18/)
- [Color event sequences from the CED dataset Scheerlinck et al., CVPR'18](http://rpg.ifi.uzh.ch/data/E2VID/datasets/CED_CVPRW19/)

## Working with ROS

Because PyTorch recommends Python 3 and ROS is only compatible with Python2, it is not straightforward to have the PyTorch reconstruction code and ROS code running in the same environment.
To make things easy, the reconstruction code we provide has no dependency on ROS, and simply read events from a text file or ZIP file.
We provide convenience functions to convert ROS bags (a popular format for event datasets) into event text files.
In addition, we also provide scripts to convert a folder containing image reconstructions back to a rosbag (or to append image reconstructions to an existing rosbag).

**Note**: it is **not** necessary to have a sourced conda environment to run the following scripts. However, [ROS](https://www.ros.org/) needs to be installed and sourced.

### rosbag -> events.txt

To extract the events from a rosbag to a zip file containing the event data:

```bash
python scripts/extract_events_from_rosbag.py /path/to/rosbag.bag \
  --output_folder=/path/to/output/folder \
  --event_topic=/dvs/events
```

### image reconstruction folder -> rosbag

```bash
python scripts/image_folder_to_rosbag.py \
  --datasets dynamic_6dof \
  --image_folder /path/to/image/folder \
  --output_folder /path/to/output_folder \
  --image_topic /dvs/image_reconstructed
```

### Append image_reconstruction_folder to an existing rosbag

```bash
cd scripts
python embed_reconstructed_images_in_rosbag.py \
  --rosbag_folder /path/to/rosbag/folder \
  --datasets dynamic_6dof \
  --image_folder /path/to/image/folder \
  --output_folder /path/to/output_folder \
  --image_topic /dvs/image_reconstructed
```

### Generating a video reconstruction (with a fixed framerate)

It can be convenient to convert an image folder to a video with a fixed framerate (for example for use in a video editing tool).
You can proceed as follows:

```bash
export FRAMERATE=30
python resample_reconstructions.py -i /path/to/input_folder -o /tmp/resampled -r $FRAMERATE
ffmpeg -framerate $FRAMERATE -i /tmp/resampled/frame_%010d.png video_"$FRAMERATE"Hz.mp4
```

## Acknowledgements

This code borrows from the following open source projects, whom we would like to thank:

- [pytorch-template](https://github.com/victoresque/pytorch-template)
