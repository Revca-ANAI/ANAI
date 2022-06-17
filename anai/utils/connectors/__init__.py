import os

import boto3
import modin.pandas as pd
import sqlalchemy
import yaml
from anai.utils.connectors.data_handler import df_loader
from colorama import Fore
from google.cloud import bigquery
from google.oauth2 import service_account
from snowflake.sqlalchemy import URL


def connect_to_sql(
    db_name,
    db_user,
    db_password,
    db_host,
    table_name,
    db_driver="mysql+pymysql",
    ssl_args=None,
    suppress=None,
):
    """
    Connect to a SQL database.

    Parameters
    ----------
    db_name : str
        Name of the database.
    db_user : str
        Username for the database.
    db_password : str
        Password for the database.
    db_host : str
        Hostname for the database.
        if port other than 33060 is used, the hostname should be in the form of 'hostname:port'.
    db_port : str
        Port for the database.
    db_driver : str
        Driver for the database.
        default: mysql+pymysql

    Returns
    -------
    df : pandas.DataFrame
    """
    print(Fore.YELLOW + "Loading Data from SQL [*]\n") if not suppress else None
    engine = sqlalchemy.create_engine(
        f"{db_driver}://{db_user}:{db_password}@{db_host}/{db_name}",
        connect_args=ssl_args,
    )
    inspector = sqlalchemy.inspect(engine)
    tables = inspector.get_table_names()
    if table_name not in tables:
        raise ValueError(f"Table {table_name} not found in database {db_name}.")
    df = pd.read_sql_table(table_name, engine)
    if df is not None:
        print(
            Fore.GREEN + "Data Loaded Successfully [", "\u2713", "]\n"
        ) if not suppress else None
    return df


def connect_to_bigquery(project_id, credentials_path, dataset, table, suppress=None):
    print(Fore.YELLOW + "Loading Data from BigQuery [*]\n") if not suppress else None
    rows_val = []
    rows_key = []
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path
    )
    client = bigquery.Client(credentials=credentials, project=project_id, location="US")
    query_job = client.query(
        """
   SELECT *
   FROM {0}.{1}.{2}
   """.format(
            project_id, dataset, table
        )
    )
    results = query_job.result()
    for row in results:
        rows_val.append(row.values())
        rows_key.append(list(row.keys()))
    df = pd.DataFrame(rows_val, columns=rows_key[0])
    if df is not None:
        print(
            Fore.GREEN + "Data Loaded Successfully [", "\u2713", "]\n"
        ) if not suppress else None
    return df


def connect_to_s3(
    aws_access_key_id,
    aws_secret_access_key,
    bucket_name,
    file_name,
    region_name=None,
    suppress=None,
):
    print(Fore.YELLOW + "Loading Data from S3 Bucket[*]\n") if not suppress else None
    client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    clientResponse = client.list_buckets()

    buckets = [bucket["Name"] for bucket in clientResponse["Buckets"]]
    if bucket_name not in buckets:
        raise ValueError(f"Bucket {bucket_name} not found.")

    obj = client.get_object(Bucket=bucket_name, Key=file_name)
    df = df_loader(obj=obj, objfilepath=file_name)
    if df is not None:
        print(
            Fore.GREEN + "Data Loaded Successfully [", "\u2713", "]\n"
        ) if not suppress else None
    return df


def parse_config(config):
    fin_config = {}
    for i in range(len(config)):
        for j, k in config[i].items():
            fin_config[j] = k
    config = fin_config
    return config


def get_config(config_path, i=0):

    if os.path.exists(config_path):
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                if not config is None:
                    return [
                        parse_config(config[list(config.keys())[i]]),
                        list(config.keys())[i],
                    ]
                else:
                    return None
            except yaml.YAMLError as exc:
                # print(exc)
                return None
                pass
    else:
        return None


def load_data_from_config(config_path, i=0, suppress=False):
    config = get_config(config_path, i=i)
    if config[1] == "S3":
        return (
            connect_to_s3(
                aws_access_key_id=config[0]["AWS_access_key_id"],
                aws_secret_access_key=config[0]["AWS_secret_access_key"],
                bucket_name=config[0]["Bucket_name"],
                file_name=config[0]["File_name"],
                region_name=config[0]["Region_name"]
                if "Region_name" in config[0]
                else None,
                suppress=suppress,
            ),
            config[0]["Target"],
        )
    elif config[1] == "BigQuery":
        return (
            connect_to_bigquery(
                project_id=config[0]["Project_id"],
                credentials_path=config[0]["Credentials"],
                dataset=config[0]["Database_name"],
                table=config[0]["Table_name"],
                suppress=suppress,
            ),
            config[0]["Target"],
        )

    elif config[1] == "SQL":
        return (
            connect_to_sql(
                db_name=config[0]["Database_name"],
                db_user=config[0]["Username"],
                db_password=config[0]["Password"],
                db_host=config[0]["Host"],
                table_name=config[0]["Table_name"],
                ssl_args={"ssl_ca": config[0]["SSL_Args"]},
                suppress=suppress,
            ),
            config[0]["Target"],
        )

    else:
        raise ValueError(
            "Invalid connector. ANAI supports S3, BigQuery and SQL connectors"
        )
