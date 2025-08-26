.. meta::
    :thumbnail: https://machine-learning-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/architecture_diagram.png
    :description: Zero-Shot Time Series Forecasting with Chronos using Amazon Bedrock and ClickHouse
    :keywords: Time Series, LLM, Forecasting, Amazon Bedrock, Amazon Chronos

.. post:: August 22, 2025
    :tags: Time Series, LLM, Forecasting, Amazon Web Services, Amazon Bedrock, Amazon Chronos
    :category: Time Series Foundation Models
    :author: Flavia
    :language: en

###############################################################################################################
Zero-Shot Time Series Forecasting with Chronos using Amazon Bedrock and ClickHouse
###############################################################################################################

.. raw:: html

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/architecture_diagram.png"
        style="width:100%"
    >

1. Overview
***************************************************************************************************************

The emergence of large language models (LLMs) with zero-shot generalization capabilities in sequence modelling
tasks has led to the development of time series foundation models (TSFMs) based on LLM architectures.
By representing time series as sequences of tokens, TSFMs can leverage LLMs' capability to extrapolate future
patterns from the context data.
TSFMs eliminate the traditional need for domain-specific model development, allowing organizations to deploy
accurate time series solutions faster.

In this post, we will focus on Chronos, a family of TSFMs for probabilistic time series forecasting
developed by Amazon.
In contrast to other TSFMs, that rely on LLMs pre-trained on text, Chronos models are trained from scratch
on a large collection of time series datasets.
Moreover, unlike other TSFMs, which require fine-tuning on in-domain data, Chronos models generate accurate
zero-shot forecasts, without any task-specific adjustments.

Recently, the Chronos family of TSFMs has been extended with Chronos-Bolt, a faster, more accurate, and more
memory-efficient Chronos model that can also be used on CPU. Chronos-Bolt is available in AutoGluon-TimeSeries,
Amazon SageMaker JumpStart and Amazon Bedrock.

In the rest of this post, we will walk through a practical example of using Chronos-Bolt with time series data
stored in ClickHouse. We will create a Bedrock endpoint, then build a Lambda function that invokes the Bedrock
endpoint with context data queried from ClickHouse and returns the forecasts.

2. Solution
***************************************************************************************************************

In this particular example, we will work with the 15-minute time series of the Italian electricity system's
total demand, which we downloaded from `Terna's data portal <https://dati.terna.it/en/download-center#/load/total-load>`__
and stored in a table in ClickHouse which we called ``total_load_data``. However, given that Chronos-Bolt
doesn't require any domain adaptation, the same solution can be applied to any other time series.

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/total_load_data.png"
        style="width:40%"
    >

    <p>
    <span style="font-weight:600">Figure 1:</span>
    <span>Last 10 rows of </span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">total_load_data</span>
    </code>
    <span>ClickHouse table.</span>
    </p>

    </div>

.. note::
    To be able to run the code provided in the rest of this section, you will need to have Boto3 and the AWS-CLI installed on your machine.
    You will also need to update several variables in the code to reflect your AWS
    configuration - such as your AWS account number, region, service roles, etc. - as will be outlined below.

2.1 Create the Bedrock endpoint
===============================================================================================================

We start by deploying Chronos-Bolt to a Bedrock endpoint hosted on a CPU EC2 instance.
This can be done using `Boto3 <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/create_marketplace_model_endpoint.html>`__
as in the code below, with the `AWS-CLI <https://docs.aws.amazon.com/cli/latest/reference/bedrock/create-marketplace-model-endpoint.html>`__,
or directly from the Bedrock console.

.. note::
    If using the code below, make sure to replace the following variables:

    -  ``"<bedrock-marketplace-arn>"``: The Bedrock marketplace ARN of Chronos-Bolt (Base) model.
    -  ``"<bedrock-execution-role>"``: The Bedrock execution role ARN.

.. code:: python

   import boto3

   # Create the Bedrock client
   bedrock_client = boto3.client("bedrock")

   # Create the Bedrock endpoint
   response = bedrock_client.create_marketplace_model_endpoint(
       modelSourceIdentifier="<bedrock-marketplace-arn>",
       endpointConfig={
           "sageMaker": {
               "initialInstanceCount": 1,
               "instanceType": "ml.m5.4xlarge",
               "executionRole": "<bedrock-execution-role>"
           }
       },
       endpointName="chronos-bedrock-endpoint",
       acceptEula=True,
   )

   # Get the Bedrock endpoint ARN
   bedrock_endpoint_arn = response["marketplaceModelEndpoint"]["endpointArn"]

2.2 Create the Lambda function for invoking the Bedrock endpoint with ClickHouse data
===============================================================================================================

We now build a Lambda function for invoking the Bedrock endpoint with time series data stored in ClickHouse.

2.2.1 Create the Docker image
---------------------------------------------------------------------------------------------------------------

In order to create the Lambda function's Docker image in Elastic Container Registry (ECR), we need the following files:

- ``app.py``: The Python code of the Lambda function.
- ``requirements.txt``: The list of dependencies that need to be installed in the Docker container.
- ``Dockerfile``: The file containing the instructions to build the Docker image.

.. raw:: html

    <p>
    <span style="font-weight:600">2.2.1.1</span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">app.py</span>
    </code>
    </p>

The ``app.py`` Python script with the entry point of the Lambda function is reported below.
The ``handler`` function has two arguments:

- ``event``: The input payload with the request parameters.
- ``context``: The runtime information about the invocation.

In this case, the ``event`` object is expected to include the following fields:

- ``"initialization_timestamp"``: The first timestamp for which the forecasts should be generated.
- ``"frequency"``: The frequency of the time series, in number of minutes.
- ``"context_length"``: The number past time series values (prior to ``initialization_timestamp``) to use as context.
- ``"prediction_length"``: The number of future time series values (on and after ``initialization_timestamp``) to predict.
- ``"quantile_levels"``: The quantiles to be predicted at each future time step.

The ``context`` object is automatically generated at runtime and does not need to be provided.

The Lambda function connects to ClickHouse using `ClickHouse Connect <https://clickhouse.com/docs/integrations/python>`__
and loads the context data using the ``query_df`` method, which returns the query output in a Pandas Dataframe.
After that, the Lambda function invokes the Bedrock endpoint with the context data.
The Bedrock endpoint response includes the predicted mean and the predicted quantiles of the time series
at each future time step, which the Lambda function returns to the user in JSON format
together with the corresponding timestamps.

.. note::
    Before deploying the Lambda function, make sure to replace the following variables:

    -  ``"<clickhouse-host>"``: The ClickHouse host.
    -  ``"<clickhouse-user>"``: The ClickHouse username.
    -  ``"<clickhouse-password>"``: The ClickHouse password.
    -  ``"<bedrock-endpoint-arn>"``: The Bedrock endpoint ARN.

.. code:: python

   import json
   import boto3
   import pandas as pd
   import clickhouse_connect

   def handler(event, context):
       """
       Generate zero-shot forecasts with Chronos-Bolt (Base) Amazon Bedrock endpoint using data stored in ClickHouse.

       Parameters:
       ========================================================================================================
       event: dict.
           A dictionary with the following keys:

           initialization_timestamp: str.
               The initialization timestamp of the forecasts, in ISO format (YYYY-MM-DD HH:mm:ss).

           frequency: int.
               The frequency of the time series, in minutes.

           context_length: int.
               The number of past time steps to use as context.

           prediction_length: int.
               The number of future time steps to predict.

           quantile_levels: list of float.
               The quantiles to be predicted at each future time step.

       context: AWS Lambda context object, see https://docs.aws.amazon.com/lambda/latest/dg/python-context.html.
       """
       # Create the ClickHouse client
       clickhouse_client = clickhouse_connect.get_client(
           host="<clickhouse-host>",
           user="<clickhouse-user>",
           password="<clickhouse-password>",
           secure=True
       )

       # Load the input data from ClickHouse
       df = clickhouse_client.query_df(
           f"""
               select
                   timestamp,
                   total_load
               from
                   total_load_data
               where
                   timestamp < toDateTime('{event['initialization_timestamp']}')
               and
                   timestamp >= toDateTime('{event['initialization_timestamp']}') - INTERVAL {int(event['frequency']) * int(event['context_length'])} MINUTES
               order by
                   timestamp asc
           """
       )

       # Create the Bedrock client
       bedrock_runtime_client = boto3.client(
           service_name="bedrock-runtime"
       )

       # Invoke the Bedrock endpoint with the ClickHouse data
       response = bedrock_runtime_client.invoke_model(
           modelId="<bedrock-endpoint-arn>",
           body=json.dumps({
               "inputs": [{
                   "target": df["total_load"].values.tolist(),
               }],
               "parameters": {
                   "prediction_length": event["prediction_length"],
                   "quantile_levels": event["quantile_levels"],
               }
           })
       )

       # Extract the forecasts
       predictions = json.loads(response["body"].read()).get("predictions")[0]

       # Add the timestamps to the forecasts
       predictions = {
           "timestamp": [
               x.strftime("%Y-%m-%d %H:%M:%S")
               for x in pd.date_range(
                   start=event["initialization_timestamp"],
                   periods=event["prediction_length"],
                   freq=f"{event['frequency']}min",
               )
           ]
       } | predictions

       # Return the forecasts
       return {
           "statusCode": 200,
           "body": json.dumps(predictions)
       }

.. raw:: html

    <p>
    <span style="font-weight:600">2.2.1.2</span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">requirements.txt</span>
    </code>
    </p>


The ``requirements.txt`` file with the list of dependencies is as follows:

::

   boto3==1.34.84
   clickhouse_connect==0.8.18
   pandas==2.3.1

.. raw:: html

    <p>
    <span style="font-weight:600">2.2.1.3</span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">Dockerfile</span>
    </code>
    </p>


The standard ``Dockerfile`` using the Python 3.12 AWS base image for Lambda is also provided for reference:

.. code:: bash

   FROM amazon/aws-lambda-python:3.12

   COPY requirements.txt  .

   RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

   COPY app.py ${LAMBDA_TASK_ROOT}

   CMD ["app.handler"]

2.2.2 Build the Docker image and push it to ECR
---------------------------------------------------------------------------------------------------------------

When all the files are ready, we can build the Docker image and push it to ECR
with the AWS-CLI as shown in the ``build_and_push.sh`` script below.

.. note::
    Before running the script, make sure to replace the following variables:

    -  ``"aws-account-id>"``: The AWS account number.
    -  ``"<ecr-repository-region>"``: The region of the ECR repository.
    -  ``"<ecr-repository-name>"``: The name of the ECR repository.

.. code:: bash

   aws_account_id="<aws-account-id>"
   region="<ecr-repository-region>"
   algorithm_name="<ecr-repository-name>"

   aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$region.amazonaws.com

   aws ecr describe-repositories --repository-names ${algorithm_name} || aws ecr create-repository --repository-name ${algorithm_name}

   docker build -t $algorithm_name .

   docker tag $algorithm_name:latest $aws_account_id.dkr.ecr.$region.amazonaws.com/$algorithm_name:latest

   docker push $aws_account_id.dkr.ecr.$region.amazonaws.com/$algorithm_name:latest


2.2.3 Create the Lambda function from the Docker image in ECR
---------------------------------------------------------------------------------------------------------------

After the Docker image has been pushed to ECR, we can create the Lambda function using `Boto3 <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda/client/create_function.html>`__
as in the code below, with the `AWS-CLI <https://docs.aws.amazon.com/cli/latest/reference/lambda/create-function.html>`__,
or directly from the Lambda console.

.. note::
    If using the code below, make sure to replace the following variables:

    -  ``"<ecr-image-uri>"``: The URI of the ECR image with the code of the Lambda function.
    -  ``"<lambda-execution-role>"``: The Lambda execution role ARN.

.. code:: python

   import boto3

   # Create the Lambda client
   lambda_client = boto3.client("lambda")

   # Create the Lambda function
   response = lambda_client.create_function(
       FunctionName="chronos-lambda-function",
       PackageType="Image",
       Code={
           "ImageUri": "<ecr-image-uri>"
       },
       Role="<lambda-execution-role>",
       Timeout=900,
       MemorySize=128,
       Publish=True,
   )

2.3 Invoke the Lambda function and generate the forecasts
===============================================================================================================

After the Lambda function has been created, we can invoke it to generate the forecasts.

The code below defines a Python function which invokes the Lambda function with the
inputs discussed in the previous section and casts the Lambda function's JSON output
to Pandas Dataframe.

Next, the code makes two invocations: the first time it requests the forecasts over a
past time window for which historical data is already available, which allows us to assess how
close the forecasts are to the actual data, while the second time it requests the forecasts
over a future time window for which the data is not yet available.

In both cases, the Lambda function is invoked with a context window of 3 weeks to generate 1-day-ahead forecasts.

.. code:: python

   import io
   import json
   import boto3
   import pandas as pd

   def invoke_lambda_function(
       initialization_timestamp,
       frequency,
       context_length,
       prediction_length,
       quantile_levels,
       function_name
   ):
       """
       Invoke the Lambda function that generates zero-shot forecasts with Chronos-Bolt (Base)
       Amazon Bedrock endpoint using data stored in ClickHouse.

       Parameters:
       ========================================================================================================
       initialization_timestamp: str.
           The initialization timestamp of the forecasts, in ISO format (YYYY-MM-DD HH:mm:ss).

       frequency: int.
           The frequency of the time series, in minutes.

       context_length: int.
           The number of past time steps to use as context.

       prediction_length: int.
           The number of future time steps to predict.

       quantile_levels: list of float.
           The quantiles to be predicted at each future time step.

       function_name: str.
           The name of the Lambda function.
       """
       # Create the Lambda client
       lambda_client = boto3.client("lambda")

       # Invoke the Lambda function
       response = lambda_client.invoke(
           FunctionName=function_name,
           Payload=json.dumps({
               "initialization_timestamp": initialization_timestamp,
               "frequency": frequency,
               "prediction_length": prediction_length,
               "context_length": context_length,
               "quantile_levels": quantile_levels
           })
       )

       # Extract the forecasts in a data frame
       predictions = pd.read_json(io.StringIO(json.loads(response["Payload"].read())["body"]))

       # Return the forecasts
       return predictions

   # Define the Lambda function name and input parameters
   frequency = 15
   context_length = 24 * 4 * 7 * 3
   prediction_length = 24 * 4
   quantile_levels = [0.1, 0.5, 0.9]
   function_name = "chronos-lambda-function"

   # Generate the forecasts over a past time window
   predictions = invoke_lambda_function(
       initialization_timestamp="2025-08-17 00:00:00",
       frequency=frequency,
       context_length=context_length,
       prediction_length=prediction_length,
       quantile_levels=quantile_levels,
       function_name=function_name
   )

   # Generate the forecasts over a future time window
   forecasts = invoke_lambda_function(
       initialization_timestamp="2025-08-18 00:00:00",
       frequency=frequency,
       context_length=context_length,
       prediction_length=prediction_length,
       quantile_levels=quantile_levels,
       function_name=function_name
   )


.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/predictions_table.png"
        style="width:100%"
    >

    <p>
    <span style="font-weight:600">Figure 2:</span>
    <span>Last 10 rows of </span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">predictions</span>
    </code>
    <span>DataFrame.</span>
    </p>

    </div>


.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/forecasts_table.png"
        style="width:100%"
    >

    <p>
    <span style="font-weight:600">Figure 3:</span>
    <span>Last 10 rows of </span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">forecasts</span>
    </code>
    <span> DataFrame.</span>
    </p>

    </div>

2.4 Compare the forecasts to the historical data stored in ClickHouse
===============================================================================================================

Now that the forecasts have been generated, we can compare them to the historical data stored in ClickHouse.
We again use ClickHouse Connect to query the database and retrieve the results directly into a Pandas DataFrame.

.. code:: python

   import clickhouse_connect

   # Create the ClickHouse client
   clickhouse_client = clickhouse_connect.get_client(
       host="<clickhouse-host>",
       user="<clickhouse-user>",
       password="<clickhouse-password>",
       secure=True
   )

   # Load the historical data from ClickHouse
   df = clickhouse_client.query_df(
       """
       select
           timestamp,
           total_load
       from
           total_load_data
       where
           timestamp >= toDateTime('2025-08-18 23:45:00') - INTERVAL 14 DAYS
       order by
           timestamp asc
       """
   )

   # Outer join the historical data with the model outputs
   output = pd.merge(
       left=df,
       right=pd.concat([predictions, forecasts], axis=0),
       on="timestamp",
       how="outer"
   )

The results show that the forecasts are closely aligned with the actual data,
demonstrating the model's ability to generalize effectively in a zero-shot setting.
Despite a holiday occurring on the last Friday of the context window,
the model produces accurate forecasts for the subsequent Sunday
and correctly anticipates an increase in energy demand on the following Monday,
highlighting its strength in capturing complex temporal patterns.

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/forecasts_plot.png"
        style="width:100%"
    >

    <p>
    <span style="font-weight:600">Figure 4:</span>
    <span>Chronos-Bolt forecasts against historical total load data.</span>
    </p>

    </div>

