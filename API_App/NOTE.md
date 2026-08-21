# API App

> [!IMPORTANT]
> Due to project complexity and different running resources, you can find the files you ask to be placed in this folder in the following paths:

1. These are all the dockerfiles deployed on the Google Cloud Platform, to run the cloud services "app", "collector", and "retraining":

```sh
.
└── docker
    ├── Dockerfile.app
    ├── Dockerfile.etl
    └── Dockerfile.train
```

2. These scripts contains the "collector" cron job (0 */5 * * *) for the data collection and the "retraining" one cron (0 */2 * * *) for the model retraining. All the scripts starting with "data_*" are part of the ETL process, and the retraining ones starting with "train_*".

```sh
.
└── jobs
    ├── data_extract.py
    ├── data_load.py
    ├── data_transform.etl
    ├── job_collector.py
    ├── job_retraining.py
    ├── train_load.py
    └── train_model.py
```

3. This file represents the source code of the running application:

```sh
.
└── app
    └── app.py
