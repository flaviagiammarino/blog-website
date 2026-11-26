:layout: post

.. post:: November 24, 2026
    :tags: Amazon SageMaker
    :category: Foundation Models
    :author: Flavia
    :description: Deploy the MOMENT time series foundation model on Amazon SageMaker
    :keywords: Time Series, Anomaly Detection, Foundation Model, Amazon SageMaker
    :image_src: https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/aws-architecture.png
    :image_alt: Architecture diagram of a SageMaker AI model, built from a PyTorch inference image in ECR and model artifacts in S3, serving a real-time endpoint, an asynchronous endpoint and a batch transform job.
    :summary: A deployment of the pre-trained MOMENT foundation model on Amazon SageMaker for zero-shot time series anomaly detection, with real-time, asynchronous and batch inference.

###############################################################################################################
Zero-shot time series anomaly detection with the MOMENT foundation model on Amazon SageMaker
###############################################################################################################

.. raw:: html

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/aws-architecture.png"
        style="width:100%"
        alt="Architecture diagram of a SageMaker AI model, built from a PyTorch inference image in ECR and model artifacts in S3, serving a real-time endpoint, an asynchronous endpoint and a batch transform job."
    >

1. Overview
***************************************************************************************************************

Organizations that monitor sensors, devices or metrics need to detect anomalies across thousands of time series.
With standard methods, this typically requires training and maintaining a separate model for each time series, or for each group of related time series.
Time series foundation models remove this requirement, as they are pre-trained on large and diverse collections of time series,
and can be applied directly to new data.

This post shows how to deploy the MOMENT `[1] <#references>`_ foundation model on Amazon SageMaker for anomaly detection.
The model takes a time series as input and returns its reconstruction, which represents the expected normal pattern of the data.
The reconstruction error, which measures the deviation between the data and the reconstruction, can be used as an anomaly score
to detect the observations that deviate substantially from their expected values.

MOMENT generates the reconstruction zero-shot, that is, without further training or fine-tuning.
Since no training is involved, the Amazon SageMaker deployment includes only an inference image, and no training image.
The inference image supports batch, real-time and asynchronous inference on single-GPU and multi-GPU instances.

The rest of this post is organized as follows.
`Section 2 <#model-description>`_ describes the architecture of the MOMENT models.
`Section 3 <#model-deployment>`_ explains how to prepare the model artifacts and create the inference image.
`Section 4 <#model-usage>`_ demonstrates the use of the model on sample time series.
`Section 5 <#conclusion>`_ summarizes the approach and its limitations.

2. Model description
**************************************************************************************************************

MOMENT `[1] <#references>`_ is a family of open-source time series foundation models
pre-trained on a large collection of public time series datasets.
MOMENT models support multiple time series analysis tasks, such as reconstruction, forecasting and classification.
Both the source code and the weights of the MOMENT models were released by the authors under the MIT License.

The models process the time series in windows of 512 time steps.
Each window is normalized by subtracting the mean and dividing by the standard deviation.
After that, it is split into 64 non-overlapping subsequences of 8 time steps, called patches.
A linear layer maps each patch to an embedding, and the 64 embeddings are processed by a T5 encoder `[2] <#references>`_.
A task-specific head then maps the encoder output to the final prediction.
For reconstruction and forecasting tasks, the prediction is transformed back to the original scale by multiplying by the standard deviation and adding the mean.

MOMENT models are released in three sizes, small, base and large, which differ in the size of the encoder.
This post uses MOMENT-1-large, whose encoder has the Flan-T5-large architecture `[3] <#references>`_,
with 1024-dimensional embeddings and a stack of 24 blocks, each made of a self-attention layer followed by a feed-forward network.
The self-attention layer has 16 heads of 64 units each, resulting in 1024 units in total, and uses fully visible masking:
each patch attends to all the patches in the window, both before and after it, so that its representation takes into account the context on both sides.
The feed-forward network has 2816 hidden units and a gated GELU activation.

Pre-training uses masked reconstruction: patches are masked uniformly at random,
and the model is trained to reconstruct them from the remaining patches.
The reconstruction head is pre-trained together with the encoder, so the model can be used zero-shot for this task.
The forecasting and classification heads are not pre-trained.

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/model-architecture.png"
        style="width:100%;"
        alt="Architecture of the MOMENT-1-large reconstruction model."
    >

    <p>
    <span style="font-weight:600">Figure 1:</span>
    <span>Architecture of the MOMENT-1-large reconstruction model.</span>
    </p>

    </div>

3. Model deployment
***************************************************************************************************************

The Amazon SageMaker model is built from two components: the model artifacts stored in Amazon S3 and an inference image stored in Amazon ECR.
The S3 bucket with the model artifacts and the ECR repository with the inference image must be in the same AWS region.

To follow along, you will need Docker and the AWS CLI installed on your machine, and an IAM user or role with permissions to create ECR repositories, push images to ECR and upload objects to S3.

3.1 Model artifacts
===============================================================================================================

The model artifacts are the configuration (``config.json``) and the weights (``model.safetensors``) of the pre-trained MOMENT-1-large checkpoint,
which are downloaded from `Hugging Face <https://huggingface.co/AutonLab/MOMENT-1-large>`__ (commit ``ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc``),
packaged into a ``model.tar.gz`` file and stored in an S3 bucket.

SageMaker downloads the model artifacts from S3 and extracts them in the container when the model starts.
This step does not require internet access, so the model can be deployed with network isolation enabled.

3.2 Inference image
===============================================================================================================

The inference image is built from four files, which are provided in this post's `GitHub repository <https://github.com/flaviagiammarino/sagemaker-moment-anomaly-detection/tree/main/container>`__.

.. code:: text

    container/
    ├── Dockerfile
    ├── build_and_push.sh
    ├── inference.py
    └── modules.py

``Dockerfile`` extends the AWS Deep Learning Container for PyTorch 2.5.1 GPU inference (Python 3.11, CUDA 12.4)
with the model code (``modules.py``) and the inference handler (``inference.py``).
``build_and_push.sh`` builds the image and pushes it to ECR.

3.2.1 Model code
---------------------------------------------------------------------------------------------------------------

The model code (``modules.py``) is a reduced version of the `original source code <https://github.com/moment-timeseries-foundation-model/moment>`__.
The forecasting and classification tasks are dropped, together with their heads, as the model is used for reconstruction only.
The code for pre-training and fine-tuning, including the mask that hides patches during pre-training, is removed, as the model is loaded with frozen weights and is used for inference only.

The mask that marks missing and padded values is also removed.
The original source code does not support missing values in the ``detect_anomalies`` method. It does support padding for
time series shorter than 512 time steps, but the paper `[1] <#references>`_ does not explicitly discuss the quality of the model's reconstructions
on shorter time series, where less context is available.
This implementation therefore requires each time series to have at least 512 time steps and no missing values.

Unlike the original source code, normalization is implemented as a function that returns the mean and standard deviation of each window, instead of a class that stores them as attributes, to make sure they are never overwritten by a concurrent request.

3.2.2 Inference handler
---------------------------------------------------------------------------------------------------------------

The inference handler (``inference.py``) replaces the default functions for loading the model (``model_fn``), decoding the request (``input_fn``), running the model (``predict_fn``) and encoding the response (``output_fn``).
The inference handler expects requests with content type ``text/csv``, and returns responses in the same format.
When the container starts, ``model_fn`` loads the model and moves it to the GPU.
When the container receives a request, ``predict_fn`` runs the model to generate the reconstructions of the input time series.

``predict_fn`` splits each time series into overlapping windows of 512 time steps,
with a stride of 128 time steps. The last window ends at the end of the time series, so that no window needs to be padded.
The windows of all the time series are stacked together and passed to the model in batches of 256 windows.
A batch can contain windows of different time series, as the model processes each window independently.
For each window, the model returns a reconstruction of the same length.

As the windows overlap, most time steps are reconstructed several times, at different positions in the window.
``predict_fn`` averages the reconstructions of each time step across all the windows that contain it.
This makes the final reconstruction less dependent on the position of the time step in the window,
especially for time steps near the edges of the window, where there is context on one side only.

The reconstructions generated with this implementation were compared with those of the original code on the sample data in `Section 4 <#model-usage>`_, and the largest absolute difference was found to be less than :math:`10^{-5}`.

4. Model usage
**************************************************************************************************************

The notebook provided in this post's
`GitHub repository <https://github.com/flaviagiammarino/sagemaker-moment-anomaly-detection/blob/main/sample_notebook.ipynb>`__
walks through the usage of the model for batch transform, real-time and asynchronous inference.
Inference is performed on a sample dataset of three synthetic time series of 1,000 half-hourly observations each;
these are used for illustration only, to show the format of the model inputs and outputs.

Running the notebook requires an IAM role or user with permissions to create and invoke SageMaker resources, and to read, write, list and delete objects in the relevant S3 buckets.

4.1 Setup
==============================================================================================================
The notebook uses version 3.30.0 of the SageMaker Python SDK.
At first, it sets up the SageMaker session and the variables used in the rest of the code.
The SageMaker resources are created in the AWS region of the session, which must be the same as the region of the S3 bucket and the ECR repository.

.. code:: python

    import io
    import time
    import datetime
    import pandas as pd
    from sagemaker.core.helper.session_helper import Session, get_execution_role
    from sagemaker.core.resources import Model, TransformJob, EndpointConfig, Endpoint
    from sagemaker.core.shapes import (
        ContainerDefinition,
        TransformInput,
        TransformDataSource,
        TransformS3DataSource,
        TransformOutput,
        TransformResources,
        ProductionVariant,
        AsyncInferenceConfig,
        AsyncInferenceOutputConfig,
    )

    # SageMaker session
    session = Session()

    # SageMaker execution role
    role = get_execution_role()

    # S3 bucket for the input data and the reconstructions
    bucket = "<s3-bucket>"

    # EC2 instance type for the batch transform job and the endpoints
    instance_type = "<instance-type>"

    # ECR URI of the inference image
    ecr_uri = "<ecr-uri>"

    # S3 URI of the model artifacts
    s3_uri = "<s3-uri>"

    # Suffix for the names of the model, the batch transform job and the endpoints
    suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

The placeholders are set as follows:

- ``<s3-bucket>`` is the S3 bucket where the input data and the reconstructions are stored.
  It can alternatively be set to the SageMaker default bucket with ``session.default_bucket()``.
- ``<instance-type>`` is the instance type of the batch transform job and the endpoints, for instance ``ml.g4dn.xlarge``; the AWS account needs sufficient quotas for the selected instance type, for both endpoints and transform jobs.
- ``<ecr-uri>`` is the ECR URI of the inference image.
- ``<s3-uri>`` is the S3 URI of the ``model.tar.gz`` file.

The variable ``role`` is the SageMaker execution role, which SageMaker uses to run the model.
It needs read and write access to the S3 bucket and permission to pull images from the ECR repository.
When the notebook runs on a SageMaker notebook instance, ``get_execution_role()`` returns the IAM role attached to the notebook instance; elsewhere, ``role`` must be set to the ARN of the role.

The variable ``suffix`` adds the current date and time to the names of the model, the batch transform job and the endpoints, as SageMaker requires these names to be unique within the AWS account and the region.

4.2 Inference inputs
==============================================================================================================

As a next step, the notebook loads the sample data provided in this post's `GitHub repository <https://github.com/flaviagiammarino/sagemaker-moment-anomaly-detection/blob/main/sample_data/data.csv>`__,
converts it to CSV text, and uploads it to S3.
The batch transform job and asynchronous endpoint use the S3 URI (the ``data`` variable in the code below) as input,
while the real-time endpoint uses the CSV text (the ``request`` variable in the code below) directly.

.. code:: python

    # Load the sample data
    dataset = pd.read_csv("sample_data/data.csv", parse_dates=["timestamp"])

    # Convert the data to CSV text
    request = dataset.to_csv(index=False)

    # Upload the CSV text to S3
    data = session.upload_string_as_file_body(
        body=request,
        bucket=bucket,
        key="sample_data/data.csv"
    )

The input CSV contains one row per observation, with three columns: ``id``, the string identifier of the time series, ``timestamp``, the time of the observation, and ``value``, the observed value.
The timestamps are used only to sort the observations; the time series are assumed to be regularly sampled.
Multiple time series can be included in the same file, as rows with different ``id`` values.

.. code:: text

    id,timestamp,value
    A,2026-01-01 00:00:00,26
    A,2026-01-01 00:30:00,28
    ...
    B,2026-01-01 00:00:00,45
    B,2026-01-01 00:30:00,45
    ...
    C,2026-01-01 00:00:00,69
    C,2026-01-01 00:30:00,61
    ...

4.3 Model creation
==============================================================================================================

After that, the notebook creates the model, which combines the inference image in ECR with the model artifacts in S3.

.. code:: python

    model = Model.create(
        model_name=f"moment-model-{suffix}",
        primary_container=ContainerDefinition(
            image=ecr_uri,
            model_data_url=s3_uri,
            environment={"SAGEMAKER_MODEL_SERVER_WORKERS": "1"},
        ),
        execution_role_arn=role,
        enable_network_isolation=True,
    )

4.4 Batch transform
==============================================================================================================

A batch transform job starts a temporary instance, sends the content of the file to the container as one request,
and writes the response to S3, in a file with the same name as the input file and the ``.out`` extension.
The input file is not split into smaller requests, as each request must contain complete time series.
Batch transform jobs accept payloads of up to 100 MB.
This option fits periodic processing, for instance a daily or weekly scan of the time series.

The batch transform job reads the input data from S3 and writes the reconstructions to S3.

.. code:: python

    # Create the batch transform job
    transform_job = TransformJob.create(
        transform_job_name=f"moment-transform-{suffix}",
        model_name=model.model_name,
        transform_input=TransformInput(
            data_source=TransformDataSource(
                s3_data_source=TransformS3DataSource(s3_data_type="S3Prefix", s3_uri=data)
            ),
            content_type="text/csv",
        ),
        transform_output=TransformOutput(
            s3_output_path=f"s3://{bucket}/batch_predictions",
            accept="text/csv",
        ),
        transform_resources=TransformResources(
            instance_type=instance_type,
            instance_count=1,
        ),
    )

    # Wait for the batch transform job to complete
    session.sagemaker_client.get_waiter("transform_job_completed_or_stopped").wait(
        TransformJobName=transform_job.transform_job_name
    )

    # Load the reconstructions from S3
    batch_predictions = session.read_s3_file(
        bucket=bucket,
        key_prefix="batch_predictions/data.csv.out"
    )

    batch_predictions = pd.read_csv(
        io.StringIO(batch_predictions),
        parse_dates=["timestamp"]
    )

4.5 Real-time inference
==============================================================================================================

A real-time endpoint keeps the model running on a persistent instance.
This option fits applications that need the result immediately, with payloads of up to 25 MB and processing times of up to 60 seconds.

The real-time endpoint is invoked with the same CSV text that was uploaded to S3, and returns CSV text in the response.

.. code:: python

    # Create the endpoint configuration
    endpoint_config = EndpointConfig.create(
        endpoint_config_name=f"moment-endpoint-config-{suffix}",
        production_variants=[
            ProductionVariant(
                variant_name="AllTraffic",
                model_name=model.model_name,
                instance_type=instance_type,
                initial_instance_count=1,
            )
        ],
    )

    # Create the endpoint
    endpoint = Endpoint.create(
        endpoint_name=f"moment-endpoint-{suffix}",
        endpoint_config_name=endpoint_config.endpoint_config_name,
    )

    # Wait for the endpoint to be in service
    session.sagemaker_client.get_waiter("endpoint_in_service").wait(
        EndpointName=endpoint.endpoint_name
    )

    # Invoke the endpoint with the CSV text
    response = endpoint.invoke(
        body=request,
        content_type="text/csv",
        accept="text/csv",
    )

    # Extract the reconstructions from the response
    real_time_predictions = pd.read_csv(
        io.StringIO(response.body.read().decode()),
        parse_dates=["timestamp"]
    )

    # Delete the endpoint and the endpoint configuration
    endpoint.delete()
    endpoint_config.delete()

4.6 Asynchronous inference
==============================================================================================================

An asynchronous endpoint queues the request, passes its contents to the container, and writes the response to S3.
The invocation returns immediately, with the S3 location where the response will be written.
This option fits larger payloads (up to 1 GB) and longer processing times (up to one hour),
and the endpoint can be scaled down to zero instances when there are no requests.

The endpoint is invoked with the S3 URI of the same file used by the batch transform job.
The S3 location of the response is specified in the endpoint configuration.

.. code:: python

    # Create the endpoint configuration
    async_endpoint_config = EndpointConfig.create(
        endpoint_config_name=f"moment-async-endpoint-config-{suffix}",
        production_variants=[
            ProductionVariant(
                variant_name="AllTraffic",
                model_name=model.model_name,
                instance_type=instance_type,
                initial_instance_count=1,
            )
        ],
        async_inference_config=AsyncInferenceConfig(
            output_config=AsyncInferenceOutputConfig(
                s3_output_path=f"s3://{bucket}/async_predictions",
                s3_failure_path=f"s3://{bucket}/async_failures",
            )
        ),
    )

    # Create the endpoint
    async_endpoint = Endpoint.create(
        endpoint_name=f"moment-async-endpoint-{suffix}",
        endpoint_config_name=async_endpoint_config.endpoint_config_name,
    )

    # Wait for the endpoint to be in service
    session.sagemaker_client.get_waiter("endpoint_in_service").wait(
        EndpointName=async_endpoint.endpoint_name
    )

    # Invoke the endpoint with the S3 URI of the CSV file
    async_response = async_endpoint.invoke_async(
        input_location=data,
        content_type="text/csv",
        accept="text/csv",
    )

    # Wait for the reconstructions, or for the error, to be written to S3
    def s3_exists(s3_uri):
        bucket_name, key = s3_uri.removeprefix("s3://").split("/", 1)
        return key in session.list_s3_files(bucket=bucket_name, key_prefix=key)

    while not s3_exists(async_response.output_location):
        if s3_exists(async_response.failure_location):
            raise RuntimeError(f"Asynchronous inference failed, see {async_response.failure_location}")
        time.sleep(10)

    # Load the reconstructions from S3
    async_predictions = session.read_s3_file(
        bucket=bucket,
        key_prefix=async_response.output_location.removeprefix(f"s3://{bucket}/")
    )

    async_predictions = pd.read_csv(
        io.StringIO(async_predictions),
        parse_dates=["timestamp"]
    )

    # Delete the endpoint and the endpoint configuration
    async_endpoint.delete()
    async_endpoint_config.delete()

4.7 Inference outputs
==============================================================================================================

All three inference options return the input data, sorted by ``id`` and ``timestamp``, with an additional ``reconstruction`` column containing the model output. The reconstructions are the same across the three options, up to a largest absolute difference of :math:`1.5 \times 10^{-5}`.

.. code:: text

    id,timestamp,value,reconstruction
    A,2026-01-01 00:00:00,26,21.56
    A,2026-01-01 00:30:00,28,22.41
    ...
    B,2026-01-01 00:00:00,45,40.67
    B,2026-01-01 00:30:00,45,41.46
    ...
    C,2026-01-01 00:00:00,69,61.50
    C,2026-01-01 00:30:00,61,62.42
    ...

As shown in `Figure 2 <#figure-2>`_, the reconstructions follow the daily and weekly cycles of the time series and do not reproduce the noise,
so the largest reconstruction error occurs at the anomaly in each time series.

- Time series A contains a global point anomaly, a single value far above the range of the rest of the time series.
  The reconstruction stays within the normal range of the time series, while the observed value is more than twice the largest normal value.
- Time series B contains a contextual point anomaly, a single value which is within the range of the time series, but is too high for the time of the day at which it occurs.
  The reconstruction is at the bottom of the daily cycle, while the observed value is close to the top of the daily cycle.
- Time series C contains a subsequence anomaly, an interval in which the fluctuations around the daily cycle are much larger than in the rest of the time series.
  The reconstruction follows the daily cycle, while the observed values fluctuate widely around it.

.. raw:: html

    <div id="figure-2" style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/model-results.png"
        style="width:100%;"
        alt="Three synthetic time series and their MOMENT reconstructions. The shaded area marks the 24 hours around the largest squared reconstruction error of each time series."
    >

    <p>
    <span style="font-weight:600">Figure 2:</span>
    <span>Three synthetic time series and their MOMENT reconstructions. The shaded area marks the 24 hours around the largest squared reconstruction error of each time series.</span>
    </p>

    </div>

This example is not an evaluation of the model's accuracy,
which is assessed on benchmark datasets in the original paper `[1] <#references>`_.

4.8 Cleanup
===============================================================================================================

The endpoints and their configurations are deleted at the end of each inference option.
The model does not incur charges, but it can be deleted when it is no longer needed.

.. code:: python

    # Delete the model
    model.delete()

The input data and the reconstructions remain in S3 and incur storage charges until they are deleted.
The code below deletes the objects written in this example, and leaves the model artifacts in place.

.. code:: python

    import boto3

    # Delete the input data and the reconstructions from S3
    s3_bucket = boto3.resource("s3").Bucket(bucket)
    for prefix in ["sample_data/", "batch_predictions/", "async_predictions/", "async_failures/"]:
        s3_bucket.objects.filter(Prefix=prefix).delete()

The model artifacts in S3 and the inference image in ECR can be kept to create the model again, or deleted from the S3 and ECR consoles.

5. Conclusion
***************************************************************************************************************

This post showed how to deploy the pre-trained MOMENT-1-large model on Amazon SageMaker
and how to use it for zero-shot time series anomaly detection.
The model weights are used as released, with no training or fine-tuning,
and the same model serves batch transform jobs, real-time endpoints and asynchronous endpoints.
As the model artifacts are stored in Amazon S3 and nothing is downloaded at runtime, the model runs with network isolation.

The model returns the reconstructions and does not flag the anomalies.
The time steps to flag are selected by comparing the reconstruction errors with an anomaly threshold,
which still needs to be calibrated on a period of the time series that is considered normal,
for instance as a high percentile of the reconstruction errors in that period.

The approach has three main practical limitations.
First, the most recent time steps are reconstructed from past observations only,
so their reconstructions are less reliable and can change when new data becomes available.
Second, each time series is reconstructed window by window, and an anomaly that is only visible over a longer period spanning multiple windows can therefore go undetected.
Third, each time series is reconstructed independently of the others,
so an anomaly that is only visible in the relationship between several time series is not detected.

References
***************************************************************************************************************

[1] Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024).
MOMENT: A family of open time-series foundation models.
In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 16115-16152.
`doi: 10.48550/arXiv.2402.03885 <https://doi.org/10.48550/arXiv.2402.03885>`__.

[2] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020).
Exploring the limits of transfer learning with a unified text-to-text transformer.
*Journal of Machine Learning Research*, 21(140), 1-67.
`doi: 10.48550/arXiv.1910.10683 <https://doi.org/10.48550/arXiv.1910.10683>`__.

[3] Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. (2024).
Scaling instruction-finetuned language models.
*Journal of Machine Learning Research*, 25(70), 1-53.
`doi: 10.48550/arXiv.2210.11416 <https://doi.org/10.48550/arXiv.2210.11416>`__.
