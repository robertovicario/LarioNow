# API App

> [!IMPORTANT]
> Due to the complexity of the project and the different resources used to run it, the files requested for this folder are located in the following paths:

### 1. Dockerfiles

The following Dockerfiles are deployed on **Google Cloud Platform (GCP)** and are used to run the `app`, `collector`, and `retraining` services:

```sh
.
└── docker
    ├── Dockerfile.app
    ├── Dockerfile.etl
    └── Dockerfile.train
```

### 2. Scheduled Jobs

The following scripts implement the scheduled `collector` and `retraining` jobs:

- The **collector** job runs every 5 minutes: `*/5 * * * *`
- The **retraining** job runs every 2 hours: `0 */2 * * *`

All scripts starting with `data_` are part of the **ETL pipeline**, while scripts starting with `train_` are related to **model retraining**.

```text
.
└── sh
    ├── data_extract.py
    ├── data_load.py
    ├── data_transform.etl
    ├── job_collector.py
    ├── job_retraining.py
    ├── train_load.py
    └── train_model.py
```

### 3. Application Source Code

The following file contains the source code of the running application:

```text
.
└── sh
    └── app.py
```
