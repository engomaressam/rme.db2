from pyspark.sql import SparkSession
from pyspark import SparkConf, conf
from pyspark.sql.types import *
import pandas as pd
from datetime import * 
import Connections as conn
from time import * 
#import cx_Oracle
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import random 
import sys 
import pendulum
import mysql.connector as mysql
import os
import subprocess

def spark_app(APPNAME=None,Memory=None,Cores=None,DB=None) :  
        jdbc_jars = f"{conn.oraclesparkjar},{conn.mysqlsparkjar}"  
        ''' spark = SparkSession.builder.appName (APPNAME) \
                .config ("spark.master", 'spark://10.10.11.242:7077') \
                .config("spark.jars", jdbc_jars) \
                .config ("spark.executor.memory",Memory) \
                .config("spark.driver.memory", "2g") \
                .config ("spark.executor.cores",Cores) \
                .config ("spark.cores.max",Cores) \
                .config("spark.executor.instances", "2") \
                .config("spark.dynamicAllocation.enabled", "true") \
                .config("spark.dynamicAllocation.minExecutors", "1") \
                .config("spark.dynamicAllocation.maxExecutors", "10") \
                .config("spark.sql.shuffle.partitions", "100") \
                .config("spark.memory.fraction", "0.6") \
                .config("spark.memory.storageFraction", "0.5").getOrCreate ()
        spark.conf.set ("spark.sql.adaptive.coalescePartitions.enabled", "True")
        spark.conf.set ("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64")'''
        
        '''sc = SparkConf().setAppName (APPNAME) \
                .set ("spark.storage.memoryFraction", "0") \
                .set ("spark.sql.inMemoryColumnarStorage.compressed", "True") \
                .set ("spark.dynamicAllocation.enabled", "false") \
                .set ("spark.sql.inMemoryColumnarStorage.batchSize", "10000")'''
        spark = SparkSession.builder.appName (APPNAME) \
                .config ("spark.master", 'spark://10.10.11.242:7077') \
                .config("spark.jars", jdbc_jars) \
                .config ("spark.executor.memory",Memory) \
                .config ("spark.executor.cores",Cores) \
                .config ("spark.cores.max",Cores) \
                .config("spark.sql.decimalOperations.allowPrecisionLoss", "true") \
                .config("spark.executor.failOnError", "true") \
                .config("spark.yarn.maxAppAttempts", "1") \
                .config ("spark.debug.maxToStringFields", "5000").getOrCreate ()
        spark.conf.set ("spark.sql.adaptive.coalescePartitions.enabled", "True")
        spark.conf.set ("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64")
        '''config("spark.executor.instances", "4") 
        config("spark.sql.shuffle.partitions", "20") \
        config("spark.sql.datetime.java8API.enabled", "true") 
                .config("spark.sql.session.timeZone", "UTC") 
                .config("spark.sql.legacy.timeParserPolicy", "LEGACY") '''
        return spark


def connection(spark=None,newdfname=None,DB=None,
Query=None,TempView=None,type=None):

    connection_type = 'jdbc'
    db_name = DB
    if type=='UNIFIER'.upper():
            driver = 'oracle.jdbc.driver.OracleDriver'
            url = '{}:{}:thin:@//{}/{}'
            jdbcType = conn.jdbcType_oracle
            host = '10.0.16.10'
            username = conn.unifier_username
            password = conn.unifier_password
            # Set Oracle-specific fetch size to optimize data retrieval
            #fetch_size = 100

    elif type=='ERP'.upper():
            driver = 'oracle.jdbc.driver.OracleDriver'
            url = '{}:{}:thin:@//{}:1521/{}?currentSchema=RME_DEV'
            jdbcType = conn.jdbcType_oracle
            host = '10.0.11.59'
            username = conn.ERP_username
            password = conn.ERP_password
            #fetch_size = 100  # Oracle fetch size

    elif type =='MySQL':
            driver = 'com.mysql.cj.jdbc.Driver'
            url = '{}:{}://{}/{}'
            jdbcType = conn.jdbcType_mysql
            host = '10.10.11.242'
            username = conn.mysql_username
            password = conn.mysql_password

    newdfname = spark.read.format("jdbc"). \
    option("url",url.format(connection_type,jdbcType,host,db_name)). \
    option("driver", driver). \
    option("useUnicode", "true"). \
    option("continueBatchOnError", "true"). \
    option("useSSL", "false"). \
    option("user", username). \
    option("password", password). \
    option("dbtable", Query). \
    load()
    #option("fetchsize", fetch_size). \
    #option("sessionInitStatement", "ALTER SESSION SET CURRENT_SCHEMA = APPS"). \
    #repartitioned = newdfname.repartition(6)
    newdfname.createOrReplaceTempView(str(TempView))
    return newdfname
    '''# Establish JDBC connection options
    jdbc_options = {
        "url": url.format(connection_type, jdbcType, host, db_name),
        "driver": driver,
        "useUnicode": "true",
        "continueBatchOnError": "true",
        "useSSL": "false",
        "user": username,
        "password": password,
        "dbtable": Query
    }
    if type in ['UNIFIER', 'ERP']:
        jdbc_options["fetchsize"] = str(fetch_size)  # Oracle-specific fetch size
    # Read data into DataFrame
    newdfname = spark.read.format("jdbc").options(**jdbc_options).load()
    
    # Repartition DataFrame for better load balancing in the pipeline
    repartitioned = newdfname.repartition(6)  # Adjust based on system capacity
    
    # Create temporary view
    repartitioned.createOrReplaceTempView(str(TempView))    

    return repartitioned'''


def Transformation(spark,FinalQuery,*argv):
        finaldf = spark.sql(FinalQuery)
        return finaldf


def WriteFunction(finaldf,load_connection_string,tablename,load_mode,load_db_user,load_db_pass):
     #batch_size = 1000  # Set batch size for MySQL writes
     #repartitioned_df = finaldf.repartition(6)   
     finaldf.write.jdbc(url=load_connection_string, table=tablename, mode=load_mode, properties={"user": load_db_user, "password": load_db_pass,"driver" :'com.mysql.cj.jdbc.Driver'})


def indexing(DBB,table,char,*args):

 db = mysql.connect(
        host = "10.10.11.242",
        user = conn.mysql_username,
        passwd = conn.mysql_password,database = DBB
    )
 cursor = db.cursor()
   
 for arg in args :
    operation = """
           ALTER TABLE """+DBB+"""."""+table+""" modify column """+arg+""" varchar("""+char+""");
           ALTER TABLE """+DBB+"""."""+table+""" ADD INDEX ("""+arg+""")"""
    operation = filter(None, operation.split(';'))

    for i in operation:
            cursor.execute(i.strip() + ';')
            db.commit()

def Truncation(DBB,table):
        db = mysql.connect(
         host = "10.10.11.242",
        user = conn.mysql_username,
        passwd = conn.mysql_password,database = DBB
    )
        cursor = db.cursor()
        operation = """
    truncate table """+DBB+"""."""+table+""";
    """
        cursor.execute(operation)
        db.commit()
        
def Deleting(DBB,table,deletequery):
        db = mysql.connect(
         host = "10.10.11.242",
        user = conn.mysql_username,
        passwd = conn.mysql_password ,database = DBB
    )
        cursor = db.cursor()
        operation = """
    DELETE FROM """+table+""" where """+deletequery+""";
    """
        cursor.execute(operation)
        db.commit()
