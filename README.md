| <img src="docs/theme/logo.svg" width="128"> |
| - |

# LarioNow

This project is an AI-powered Python application for weather nowcasting in the Lake Como area. It combines data from a network of physical sensor stations, which collect environmental measurements every 5 minutes, with a multi-state machine learning model trained on these observations to generate short-term forecasts (30-60-90–120 minutes ahead). By leveraging high-frequency, real-time sensor data, the system aims to provide accurate hyperlocal predictions of rapidly evolving weather conditions.

## Prerequisites

> [!IMPORTANT]
>
> - uv
> - Docker

## User Interface (UI)

| <a href="#"><img src="docs/theme/ui-1.png" alt="UI" width="512"></a> | <a href="#"><img src="docs/theme/ui-2.png" alt="UI" width="512"></a> | <a href="#"><img src="docs/theme/ui-3.png" alt="UI" width="512"></a> |
| :-: | :-: | :-: |
| **Reference Station** | **Actual Measurements** | **Weather Nowcasting** |

## Instructions

Usage:

```sh
bash cmd.sh {start|stop|build|setup|collector|retraining|deploy}
```

### `build`

If you haven't built the project yet, you can do so by running:

```sh
bash cmd.sh build [--dev|--app]
```

To run test workspace such as notebooks and the projects' resources, you can use the `--dev` flag. To build the application put in the production server, you can use the `--app` flag.

> [!WARNING]
> To perform some operations, it could be necessary to authenticate with Google Cloud using your Google account by running sequentially in the terminal the two following commands:

```sh
gcloud auth login
```

```sh
gcloud auth application-default login
```

### `start`

The application can be started by running the following command:

```sh
bash cmd.sh start
```

### `stop`

To stop the program, simply run:

```sh
bash cmd.sh stop
```

### `collector`

To collect new data, you can run the following command:

```sh
bash cmd.sh collector
```

It exists a Google Cloud Run job scheduled to run every 5 minutes.

### `retraining`

To retrain the model, the following command can be used:

```sh
bash cmd.sh retraining
```

There is a Google Cloud Run job scheduled to run every hour to retrain the model with the latest data.

### `deploy`

The jobs running on the server could be deployed by running the following command after every update:

```sh
bash cmd.sh deploy
```

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
