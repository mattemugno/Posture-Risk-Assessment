# Posture Risk Assessment

Local Flask application for assessing shoulder and trunk posture from a live webcam. MediaPipe Holistic detects body landmarks; geometric angle calculations and configured thresholds determine the posture status. The repository does not train a custom risk classifier.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [Models](#models)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [Known limitations](#known-limitations)

## Introduction

Workplace ergonomics is crucial for preventing musculoskeletal disorders. This project leverages computer vision to automatically assess the risk level of different postures, helping to identify and mitigate potential health risks.

## Features

- **Posture Detection**: Detects human postures from live streaming video.
- **Joint angles**: Estimates left/right shoulder angles and trunk inclination from 2D landmarks.
- **Live feedback**: Overlays angles and colored body connections, with an optional segmentation mask.
- **Sessions**: Samples angles approximately once per second while recording and saves them in `data/*.json` when the session is stopped.
- **Statistics**: Loads a saved JSON session and displays interactive angle plots, posture status, a heatmap, and risk percentage doughnuts. `data/data_old.json` is included for trying the statistics without a camera.
- **Chart exploration**: Hover or tap to inspect values, drag to zoom, use the chart toolbar to pan, or apply a time interval with the From/To fields. The three time reports share the selected interval. Risk percentages always describe the full session, using the existing calculations.

## Installation

Use **Python 3.12.14**, pinned in `.python-version`. The project supports Python 3.12 and keeps the MediaPipe version with wheels for Linux ARM64. Dependencies are declared in `pyproject.toml` and fully pinned in `uv.lock`; `requirements.txt` is a generated export with package hashes for pip.

```sh
git clone https://github.com/mattemugno/Posture-Risk-Assessment.git
cd Posture-Risk-Assessment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements.txt
python -m pip check
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\activate.bat` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell).

If using `uv`, the equivalent setup is:

```sh
uv sync --locked
```

Installation troubleshooting:

- `No matching distribution found for mediapipe==0.10.11`: the previous pin did not provide Python 3.12 wheels. The requirements now use `mediapipe==0.10.14`, which retains the `mp.solutions` API used by the code and supports this interpreter. See the [published wheels](https://pypi.org/project/mediapipe/0.10.14/#files).
- Use only `opencv-contrib-python`, as listed in the requirements. It already includes the main OpenCV modules. Installing `opencv-python` alongside it overwrites the same `cv2` namespace; see the [OpenCV installation instructions](https://pypi.org/project/opencv-contrib-python/4.9.0.80/). If both were installed, uninstall both and reinstall the requirements.
- Check `python --version` and `python -m pip --version` to confirm that the virtual environment is active. Use `python -m pip` to install into that interpreter.
- If a Linux OpenCV import reports a missing `libGL.so.1` or GLib library, install the corresponding distribution packages (`libgl1` and `libglib2.0-0` / `libglib2.0-0t64` on Debian/Ubuntu).
- If pip cannot find even Flask, inspect its verbose output with `python -m pip install -v -r requirements.txt` for DNS, network, or package-index errors.

## Running the application

From the repository root, with the virtual environment active:

```sh
python -m src.main
```

Open <http://127.0.0.1:5000>.

1. Click **Open Video Camera** to start the webcam attached to the machine running Python (`DEVICE_ID = 0` by default). Camera capture happens on the Flask server, so a remote browser does not supply its own webcam.
2. Click **Start Session**, then **Stop Session** to save the sampled angles as JSON.
3. Select a JSON file and click **Show Statistics** to view the four reports. You can select `data_old.json` without opening the camera.
4. Close the camera when finished. Any active recording session is saved before the camera is released.

The statistics page uses a locally bundled Plotly.js library and works without a chart CDN or new Python dependencies. Each time chart provides zoom, pan, reset, and PNG download controls. **Reset zoom** restores the full interval across all time charts. The original PNG routes remain available and are displayed if interactive loading fails or JavaScript is disabled.

## Models
The application uses the legacy MediaPipe Holistic API (`mp.solutions.holistic`). `MODEL_COMPLEXITY` defaults to `0`; MediaPipe may download its lightweight pose model the first time the camera is opened, requiring internet access. The `model_path` setting for `pose_landmarker.task` is currently unused.

## Project structure

| File | Responsibility |
| --- | --- |
| `src/main.py` | Flask routes, video stream, session controls, and chart responses |
| `src/PoseManager.py` | Shared detector and evaluation lifecycle |
| `src/VideoCamera.py` | OpenCV camera capture, resizing, and frame processing |
| `src/PoseDetector.py` | MediaPipe inference, angle estimation, overlays, and JSON recording |
| `src/PostureEvaluation.py` | Loading recorded data and generating statistics |
| `src/plot_style.py` | Shared chart colors, typography, and axis styling |
| `src/video_overlay.py` | Camera angle labels, summary card, and cached font rendering |
| `src/utils.py` | Geometry, posture status rules, and drawing helpers |
| `src/config.py` | Camera, thresholds, model complexity, and paths |
| `template/` | Shared page layout, live view, and statistics HTML pages |
| `static/css/`, `static/js/` | Stylesheets and browser scripts served by Flask |
| `static/vendor/` | Pinned Plotly.js Cartesian bundle and MIT license |
| `static/fonts/` | Bundled Liberation Sans fonts and SIL Open Font License for camera text |
| `data/`, `img/` | Recorded samples and chart/placeholder images |
| `pyproject.toml`, `uv.lock`, `.python-version` | Project dependencies, locked versions, and Python interpreter |

## Configuration

Edit `src/config.py` for the camera index, model complexity, and angle thresholds. The current intermediate ranges are 21–60° for shoulders and 11–20° for the trunk; values above 60° and 20°, respectively, receive the highest risk status. Intermediate postures are also evaluated against `MAX_HOLDING_TIME = 20`.

Logging is initialized once when the Flask application is loaded. Messages go to the console and to `logs/app.log`, relative to the repository root regardless of the working directory. The file is appended across restarts and rotates at 5 MiB, retaining three backups. The default level is `INFO`; per-frame messages are logged at `DEBUG`.

To enable debug logging on Linux/macOS:

```sh
LOG_LEVEL=DEBUG python -m src.main
```

## Development

Run the regression tests with the virtual environment active:

```sh
python -m unittest discover -s tests -v
```

Change dependencies in `pyproject.toml`, update the lockfile, and regenerate the pip export together:

```sh
uv lock
uv export --locked --no-dev --no-emit-project --output-file requirements.txt
```

Commit both lockfiles. `uv sync --locked` installs the recorded versions and rejects an outdated lockfile. The `.gitignore` excludes virtual environments, Python caches, logs, and newly recorded JSON sessions, while retaining `data/data_old.json`. The frontend assets are separate from the templates; Flask generates their URLs and passes the JSON-list endpoint through an HTML data attribute.

## Known limitations

- A connected webcam is required for recording and segmentation; saved session statistics remain available without one.
- The posture status plots and risk percentages use different holding-time counting logic. Time is counted in samples rather than timestamp differences, so risk percentages need review before quantitative use.
- Samples missing any of the three angles are excluded from statistics. Interactive reports show an empty state for sessions without complete samples; the legacy PNG routes do not handle empty sessions.
- The camera and recording session are shared across browser clients in one Python process. Each statistics page retains its selected JSON file.
