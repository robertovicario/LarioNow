| <img src="docs/theme/logo.svg" width="128"> |
| - |

# LarioNow

This repository contains my project work for the Data Science for Business course at the University of Insubria, part of the MSc in Computer Science.

## Overview

This project is an AI-powered Python application for weather nowcasting in the Lake Como area. It combines data from a network of physical sensor stations, which collect environmental measurements every five minutes, with a machine learning model trained on these observations to generate short-term forecasts (30-60-90–120 minutes ahead). By leveraging high-frequency, real-time sensor data, the system aims to provide accurate hyperlocal predictions of rapidly evolving weather conditions.

## Prerequisites

> [!IMPORTANT]
>
> - uv
> - Docker

## User Interface (UI)

| <a href="#"><img src="docs/theme/cover.png" alt="UI" width="512"></a> |
| :-: |
| **Home - LarioNow** |

## Instructions

Usage:

```sh
bash cmd.sh {start|stop|setup|debug|collector|retraining|deploy_jobs}
```

### `setup`

If you haven't built the project yet, you can do so by running:

```sh
bash cmd.sh setup
```

...

```sh
bash cmd.sh debug
```

### `start`

...

```sh
bash cmd.sh start
```

### `stop`

...

```sh
bash cmd.sh stop
```

### `collector`

To collect data, you can run the following command:

```sh
bash cmd.sh collector
```

It exists a Google Cloud Run job scheduled to run every 5 minutes.

### `retraining`

To retrain the model, you can run the following command:

```sh
bash cmd.sh collector
```

It exists a Google Cloud Run job scheduled to run every 6 hours.

### `deploy_jobs`

...

## Credits

> [!WARNING]
>
> Please use this project responsibly, it was created by me for an exam session that I completed at _University of Insubria_. If you use or reference this project, please cite it as follows:
>
> ```bib
> @misc{vicario2026datascience,
>     author = {R. Vicario},
>     title  = {uninsubria-DATA_SCIENCE},
>     year   = {2026},
>     url    = {https://github.com/robertovicario/uninsubria-DATA_SCIENCE}
> }
> ```

## License

This project is distributed under [GNU General Public License version 3](https://opensource.org/license/gpl-3-0). You can find the complete text of the license in the project repository.
