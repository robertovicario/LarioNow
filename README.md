| <img src="docs/theme/logo.svg" width="128"> |
| - |

# LarioNow

This project is an AI-powered Python application for weather nowcasting in the Lake Como area. It combines data from a network of physical sensor stations, which collect environmental measurements every 5 minutes, with a multi-state machine learning model trained on these observations to generate short-term forecasts (30-60-90–120 minutes ahead). By leveraging high-frequency, real-time sensor data, the system aims to provide accurate hyperlocal predictions of rapidly evolving weather conditions.

> LarioNow is accessible online at the following link: [https://larionow-289545143980.europe-west8.run.app](https://larionow-289545143980.europe-west8.run.app)

## Prerequisites

> [!IMPORTANT]
>
> - uv
> - Docker

## User Interface (UI)

| <a href="#"><img src="docs/theme/cover.png" alt="UI" width="512"></a> |
| :-: |
| **Home - LarioNow** |

| <a href="#"><img src="docs/img/ui-1.png" alt="UI" width="512"></a> | <a href="#"><img src="docs/img/ui-2.png" alt="UI" width="512"></a> | <a href="#"><img src="docs/img/ui-3.png" alt="UI" width="512"></a> | <a href="#"><img src="docs/img/ui-4.png" alt="UI" width="512"></a> |
| :-: | :-: | :-: | :-: |
| **Reference Station** | **Actual Measurements** | **Weather Nowcasting** | **Insights** |

## Instructions

Usage:

```sh
bash cmd.sh {start|stop|build|clean|setup|collector|retraining|deploy}
```

### `setup`

If you haven't set up the project yet, you can do so by running the following command:

```sh
bash cmd.sh setup
```

A virtual environment will be created in the `.venv` folder, all the required dependencies will be installed, and a Jupyter kernel will be available for the project.

> [!WARNING]
> To perform some operations, it could be necessary to authenticate with Google Cloud using your Google account by running sequentially in the terminal the two following commands:

```sh
gcloud auth login
```

```sh
gcloud auth application-default login
```

### `build`

To make the application up and running, first you need to build the application by running the following command:

```sh
bash cmd.sh build
```

Once the build process is complete, the project will be accessible at [http://localhost:8501](http://localhost:8501).

### `start`

The Docker containers can be started by running the following command:

```sh
bash cmd.sh start
```

### `stop`

Otherwise, if you want to stop the application containers, you can run the following command:

```sh
bash cmd.sh stop
```

### `clean`

By running the following command, you can clean the project:

```sh
bash cmd.sh clean [--env|--docker]
```

If you need to clean the virtual environment, you can choose the `--env` option, while if you want to clean the Docker images, you can choose the `--docker` option.

### `collector`

To collect new data, you can run the following command:

```sh
bash cmd.sh collector
```

It exists a Google Cloud Run job scheduled to run every 5 minutes (`*/5 * * * *`).

### `retraining`

To retrain the model, the following command can be used:

```sh
bash cmd.sh retraining
```

There is a Google Cloud Run job scheduled to run every 2 hours (`0 */2 * * *`) to retrain the model with the latest data.

### `deploy`

The services and the jobs could be deployed by running the following command after every update:

```sh
bash cmd.sh deploy [--app|--jobs]
```

If you want to deploy the application, you can choose the `--app` option, while if you want to deploy the jobs, you can choose the `--jobs` option.

## Dataset

The used the dataset is a collection of environmental measurements collected from a network of physical sensor stations, property of the ***Centro Meteo Lombardo (CML)***, located in the Lake Como area. The data is collected **_every 5 minutes_**, since August 2026 starting, and includes **_various weather parameters_** such as temperature, humidity, dew point, wind speed, wind direction, pressure, and rainfall.

| <img src="docs/img/data-1.png" alt="data-1" width="512"> |
| - |
| **Figure 1:** A representation of the lakeside provinces (left) against the available meteorological station near the lake (right). |

### ETL Pipeline

To collect the data, an ***ETL pipeline*** has been implemented to extract the data from the physical sensor stations, transform it into a suitable format, and load it into a ***Google BigQuery*** table for further analysis and modeling.

> ***Extraction***

To perform the extraction, the web server of the physical sensor stations is scraped to ***retrieve the data in real-time***. The scraping, performed for each station, returns an image containing the weather parameters, as shown in ***Figure 2***.

| <img src="docs/img/data-2.png" alt="data-2" width="512"> |
| - |
| **Figure 2:** An example of what the web server of the physical sensor stations returns for a sample station. |

> ***Transformation***

The second step of the pipeline consists of transforming the image into a suitable format for further analysis and modeling. To do this, a proper ***Computer Vision algorithm*** was implemented to extract all the parameters, using an ***OCR (Optical Character Recognition) model***, which is able to recognize the text in the image and convert it into a structured format.

As it can be seen in the ***Figure 2***, there are many measurements and some additional information that could influence negatively the OCR model by introducing noise. To overcome this issue, a proper ***image pre-processing*** has been implemented to enhance the image and improve the OCR model's performance.

First of all, the image containing green and blue text: by segmenting the image into two channels (green and blue), helps to isolate the region of interest, as ***Figure 3*** shown.

| <img src="docs/img/data-2.png" alt="data-2" width="512"><img src="docs/img/data-3.png" alt="data-3" width="512"><img src="docs/img/data-4.png" alt="data-4" width="512"> |
| - |
| **Figure 3:** The first image shows the original image, while the second and third images show the segmented green and blue channels, respectively. |

For second, the image needs to be split into $N$ sub-images, where $N$ corresponds to the number of weather parameters to be extracted. Rather than applying the OCR model directly to the entire image, each parameter is first localized and isolated into a separate ***Region of Interest (ROI)***. This reduces the amount of irrelevant information provided to the OCR model and allows each measurement to be processed independently.

To identify the position of each parameter, the segmented color masks are first processed using a morphological dilation operation. This operation connects nearby pixels belonging to the same textual element, making it possible to identify each field as a single connected component. Subsequently, the contours of the resulting mask are extracted using the ***OpenCV*** contour detection algorithm. For each contour, a bounding box is computed and used to define the corresponding ROI. Very small regions are discarded as they are unlikely to contain meaningful measurements, while a small padding is added around each bounding box to preserve the complete characters at the boundaries.

**Algorithm 1:**

$$
\begin{array}{ll}
\textbf{Input:} & I, M, C \\
\textbf{Output:} & R \\[3mm]
\textbf{function}: \\
& K \leftarrow \text{RectangularKernel}(15,3) \\
& M' \leftarrow \text{Dilate}(M,K) \\
& C \leftarrow \text{FindContours}(M') \\
& R \leftarrow \emptyset \\
& \textbf{for each } c \in C \textbf{ do} \\
& \quad (x,y,w,h) \leftarrow \text{BoundingRect}(c) \\
& \quad \textbf{if } w < 15 \lor h < 15 \textbf{ then continue} \\
& \quad (x_1,y_1,x_2,y_2) \leftarrow \text{Expand}(x,y,w,h,3) \\
& \quad r \leftarrow I[y_1:y_2,x_1:x_2] \\
& \quad R \leftarrow R \cup \{(x_1,y_1,x_2,y_2,r)\} \\
& \textbf{return } R
\end{array}
$$

The final result of the transformation step is a structured dataset containing the extracted weather parameters, along with their corresponding confidence scores, as shown in ***Figure 4***.

| <img src="docs/img/data-6.png" alt="data-6" width="512"><img src="docs/img/data-7.png" alt="data-7" width="512"><img src="docs/img/data-8.png" alt="data-8" width="512"><img src="docs/img/data-9.png" alt="data-9" width="512"><img src="docs/img/data-10.png" alt="data-10" width="512"><img src="docs/img/data-11.png" alt="data-11" width="512"><img src="docs/img/data-12.png" alt="data-11" width="512"><img src="docs/img/data-13.png" alt="data-13" width="512"> |
| - |
| **Figure 4:** The image shows the final input to the OCR model, which consists of the extracted weather parameters along with their corresponding confidence scores. |

To highlight the precision of the OCR model after the transformation step, following are reported the confidence scores for each extracted parameter in the figures above:

```sh
2026-08-22 10:39:23.814 | DEBUG    | lib.utils:ocr_predict:166 -  temperature_c:   13.2 (confidence=1.000)
2026-08-22 10:39:23.883 | DEBUG    | lib.utils:ocr_predict:166 -   humidity_pct:     97 (confidence=1.000)
2026-08-22 10:39:23.954 | DEBUG    | lib.utils:ocr_predict:166 -    dew_point_c:   12.7 (confidence=0.981)
2026-08-22 10:39:24.024 | DEBUG    | lib.utils:ocr_predict:166 - wind_speed_kmh:    3.2 (confidence=0.999)
2026-08-22 10:39:24.096 | DEBUG    | lib.utils:ocr_predict:166 -       wind_dir:      N (confidence=0.994)
2026-08-22 10:39:24.180 | DEBUG    | lib.utils:ocr_predict:166 -   pressure_hpa: 1013.9 (confidence=1.000)
2026-08-22 10:39:24.251 | DEBUG    | lib.utils:ocr_predict:166 -        rain_mm:    0.3 (confidence=0.999)
2026-08-22 10:39:24.322 | DEBUG    | lib.utils:ocr_predict:166 -       rain_mmh:    0.0 (confidence=0.991)
```

To provide a more general qualitative overview of the OCR model's performance, the ***Figure 5*** shows the confidence scores for each extracted parameter over all the samples actually collected in the dataset.

| <img src="docs/img/data-14.png" alt="data-14" width="512"> |
| - |
| **Figure 6:** The image shows the confidence scores for each extracted parameter over all the samples actually collected in the dataset. |

> ***Loading***

The final stage consists of loading the structured dataset into a ***Google BigQuery*** table for further analysis and modeling.

Below, you can find the technical instructions to replicate the ETL pipeline and the data collection process.

### `collector`

To collect new data, you can run the following command:

```sh
bash cmd.sh collector
```

It exists a Google Cloud Run job scheduled to run every 5 minutes (`*/5 * * * *`).

## Results

...

### `retraining`

To retrain the model, the following command can be used:

```sh
bash cmd.sh retraining
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
