# ANAI

## Ingesting data using the inbuilt connectors

## Supported Sources

    1) MySQL
    2) S3
    3) BigQuery

## Examples

### Example Config File for MySQL

![MySql](https://revca-assets.s3.ap-south-1.amazonaws.com/mysql.png)

### Example Config File for S3

![S3](https://revca-assets.s3.ap-south-1.amazonaws.com/s3.png)

### Example Config File for BigQuery

![BigQuery](https://revca-assets.s3.ap-south-1.amazonaws.com/bigquery.png)

### Run ANAI

ANAI will automatically detect the source of the data and run the appropriate connector using the config file. When Config = True. ANAI will automatically find the anai_config.yaml file in the current working directory.

        import anai
        ai = anai.run(config=True, predictor=['rfr'])
