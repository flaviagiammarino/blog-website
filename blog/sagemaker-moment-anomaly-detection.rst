:layout: post

.. post:: November 24, 2026
    :tags: Amazon SageMaker
    :category: Foundation Models
    :author: Flavia
    :description: Zero-shot time series anomaly detection with the pre-trained MOMENT foundation model on Amazon SageMaker.
    :keywords: Time Series, Anomaly Detection, Foundation Model, MOMENT, Amazon SageMaker
    :image_src: https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/architecture_diagram.png
    :image_alt: Architecture diagram of a SageMaker AI model, built from a PyTorch inference image in ECR and model artifacts in S3, serving a real-time endpoint, an asynchronous endpoint and a batch transform job.
    :summary: A deployment of the pre-trained MOMENT foundation model on Amazon SageMaker for zero-shot time series anomaly detection, with real-time, asynchronous and batch inference.

###############################################################################################################
Zero-shot time series anomaly detection with the MOMENT foundation model on Amazon SageMaker
###############################################################################################################

.. raw:: html

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/architecture_diagram.png"
        style="width:100%"
        alt="Architecture diagram of a SageMaker AI model, built from a PyTorch inference image in ECR and model artifacts in S3, serving a real-time endpoint, an asynchronous endpoint and a batch transform job."
    >

1. Overview
***************************************************************************************************************

Reconstruction-based time series anomaly detection requires training a model to reproduce the normal behavior of the data.
The model is trained on data without anomalies, and is then used to reconstruct new data.
The reconstruction represents the expected normal pattern, and large deviations from it are flagged as anomalies.

In practice, anomaly labels are rarely available, making it difficult to select an appropriate training set.
The lack of labelled data also makes hyperparameter tuning difficult, as there is no clear metric to optimize.
Time series foundation models avoid these difficulties, as they are pre-trained on large and diverse collections of time series and can be applied to new data without further training.

This post shows how to deploy the MOMENT `[1] <#references>`_ foundation model on Amazon SageMaker for reconstruction-based time series anomaly detection.
The model is used zero-shot, that is, with the pre-trained weights and without fine-tuning.
The deployment therefore includes only an inference image, which supports real-time, asynchronous and batch inference on single-GPU and multi-GPU instances.

2. Background
**************************************************************************************************************

MOMENT `[1] <#references>`_ is a family of open-source time series foundation models
pre-trained on a large collection of public time series datasets and supporting multiple time series analysis tasks,
such as reconstruction, forecasting and classification.
Both the source code and the weights of the MOMENT models were released by the authors under the MIT License.

The models process the time series in windows of 512 time steps.
Each window is split into 64 non-overlapping subsequences of 8 time steps, called patches.
A linear layer maps each patch to an embedding, and the 64 embeddings are processed by a T5 encoder `[2] <#references>`_.
A task-specific head then maps the encoder output to the final prediction.

MOMENT models are released in three sizes, small, base and large, which differ in the size of the encoder.
This post uses MOMENT-1-large, whose encoder has the Flan-T5-large architecture `[3] <#references>`_,
with 1024-dimensional embeddings and a stack of 24 blocks, each made of a self-attention layer followed by a feed-forward network.
The self-attention layer has 16 heads of 64 units each, resulting in 1024 units in total, and uses fully visible masking:
each patch attends to all the patches in the window, both before and after it, so that its representation takes into account the context on both sides.
The feed-forward network has 2816 hidden units and a gated GELU activation.

Pre-training uses masked reconstruction: patches are masked uniformly at random, and the model is trained to reconstruct them from the remaining patches.
The reconstruction head is pre-trained together with the encoder, so the model can be used zero-shot for reconstruction, and therefore for anomaly detection and imputation.
The forecasting and classification heads are not pre-trained and must be fine-tuned.

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/model_architecture.png"
        style="width:100%;"
        alt="Architecture of the MOMENT-1-large reconstruction model."
    >

    <p>
    <span style="font-weight:600">Figure 1:</span>
    <span>Architecture of the MOMENT-1-large reconstruction model.</span>
    </p>

    </div>

3. Solution
***************************************************************************************************************

The Amazon SageMaker model is built from two components: the model artifacts stored in Amazon S3 and an inference image stored in Amazon ECR.

3.1 Model artifacts
===============================================================================================================

The model artifacts are the configuration (``config.json``) and the weights (``model.safetensors``) of the pre-trained MOMENT-1-large checkpoint,
which are downloaded from `Hugging Face <https://huggingface.co/AutonLab/MOMENT-1-large>`__,
packaged into a ``model.tar.gz`` file and stored in an S3 bucket.
This post uses commit ``ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc``.
SageMaker downloads the model artifacts from S3 and extracts them in the container when the model starts.
This step does not require internet access, so the model can be deployed with network isolation enabled.

3.2 Inference image
===============================================================================================================

The source code of the inference image is provided in this post's
`GitHub repository <https://github.com/flaviagiammarino/sagemaker-moment-anomaly-detection/tree/main/container>`__.
The image is built from four files:

.. code:: text

    container/
    ├── Dockerfile
    ├── build_and_push.sh
    ├── inference.py
    └── modules.py

``Dockerfile`` extends the AWS Deep Learning Container for PyTorch 2.5.1 GPU inference (Python 3.11, CUDA 12.4)
with the model code (``modules.py``) and the inference handler (``inference.py``),
and ``build_and_push.sh`` builds the image and pushes it to ECR.
The only dependencies added to the base container are ``transformers`` 4.54.1 and ``safetensors``.

3.2.1 Model code
---------------------------------------------------------------------------------------------------------------

``modules.py`` is a reduced version of the `original source code <https://github.com/moment-timeseries-foundation-model/moment>`__.
The forecasting and classification tasks are removed, together with their heads and with the code for pre-training and fine-tuning,
including the mask that hides patches during pre-training.
The model is loaded with frozen weights and is used for inference only.
The mask that marks padded and missing values is also removed: the container does not support them, as unobserved values cannot be classified as normal or anomalous.

3.2.2 Inference handler
---------------------------------------------------------------------------------------------------------------

The inference handler replaces the default functions for loading the model (``model_fn``), decoding the request (``input_fn``), running the model (``predict_fn``) and encoding the response (``output_fn``).

3.2.2.1 Model loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The model is loaded once, when the container starts, from the default ``/opt/ml/model`` directory with the model artifacts.
It is loaded by a single worker and is moved to the GPU.
On instances with more than one GPU, the model is replicated on all the GPUs, and each batch is split across them.

3.2.2.2 Model request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The request is CSV text with one or more univariate time series in long format, with three columns:
``id``, the identifier of the time series, ``timestamp``, the time stamp, and ``value``, the observed value.
Each time series must have at least 512 time steps and cannot contain missing or infinite values.

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

3.2.2.3 Model predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each time series is split into overlapping windows of 512 time steps, with a stride of 128 time steps.
The last window ends at the end of the time series, so that no window needs to be padded.
The windows of all the time series in the request are stacked together and passed to the model in batches of 256 windows.
A batch can contain windows of different time series, as the model processes each window independently.

For each window, the model returns its reconstruction, which has the same length as the window.
As the windows overlap, most time steps are reconstructed several times, at different positions in the window.
The reconstruction of each time step is the average of its reconstructions across the windows that contain it.
The average depends less on the position of the time step in the window,
for instance near the edges of the window, where the time step has context on one side only.

3.2.2.4 Model response
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The response is CSV text with the rows of the request and an additional column, ``reconstruction``, the reconstructed value.
The reconstructions were compared with those of the original code on the sample data,
and the largest absolute difference is less than :math:`10^{-5}`.

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

3.3 Inference options
===============================================================================================================

The same model is used for the different inference options supported by SageMaker, where the container receives the same CSV request and returns the same CSV response.
The difference is in how the request reaches the container and in where the response is delivered.

3.3.1 Batch transform
---------------------------------------------------------------------------------------------------------------

The input is a CSV file in S3.
A batch transform job starts a temporary instance, sends the content of the file to the container as one request,
and writes the response to S3, in a file with the same name and the ``.out`` extension.
The file is not split into smaller requests, as each request must contain complete time series.
The instance is stopped when the job completes.
This option fits periodic processing, for instance a daily or weekly scan of all the time series.

3.3.2 Real-time inference
---------------------------------------------------------------------------------------------------------------

The input is the CSV text in the body of the request to the endpoint, and the output is the CSV text in the body of the response.
The endpoint keeps the model running on a persistent instance.
This option fits applications that need the result immediately,
and requests that are within the payload size and response time limits of real-time endpoints.

3.3.3 Asynchronous inference
---------------------------------------------------------------------------------------------------------------

The input is a CSV file in S3, and the endpoint is invoked with the S3 location of the file.
The endpoint queues the request, passes the content of the file to the container, and writes the response to S3.
The invocation returns immediately, with the S3 location where the response will be written.
This option fits larger payloads and longer processing times,
and the endpoint can be scaled down to zero instances when there are no requests.

3.4 Example
===============================================================================================================

The notebook provided in this post's
`GitHub repository <https://github.com/flaviagiammarino/sagemaker-moment-anomaly-detection/tree/main/sample_notebook.ipynb>`__
walks through the usage of the model with the three inference options, from the upload of the data to the retrieval of the reconstructions.
In the notebook, the model is applied to three synthetic time series of 1,000 half-hourly observations each.
The three time series are synthetic, and each anomaly was designed to illustrate one type of anomaly.

- Time series A contains a global point anomaly, a single value far above the range of the rest of the time series.
- Time series B contains a contextual point anomaly, a single value which is within the range of the time series, but is too high for the time of the day at which it occurs.
- Time series C contains a subsequence anomaly, an interval in which the fluctuations around the daily cycle are much larger than in the rest of the time series.

The figure below shows the three time series with their reconstructions.
For each time series, the shaded area marks the 24 hours around the largest reconstruction error.

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/sagemaker-moment-anomaly-detection/plot.png"
        style="width:100%;"
        alt="Three synthetic time series and their MOMENT reconstructions. The highlighted windows mark the 24-hour window centred on the largest squared reconstruction error of each series."
    >

    <p>
    <span style="font-weight:600">Figure 2:</span>
    <span>Three synthetic time series and their MOMENT reconstructions. The highlighted windows mark the 24-hour window centred on the largest squared reconstruction error of each series.</span>
    </p>

    </div>

The reconstructions follow the daily and weekly cycles of the time series and do not reproduce the noise, and the largest error occurs at the anomaly.

- In time series A, the reconstruction stays within the normal range of the time series, while the observed value is more than twice the largest normal value.
- In time series B, the reconstruction is at the bottom of the daily cycle, while the observed value is close to the top of the daily cycle.
- In time series C, the reconstruction follows the daily cycle, while the observed values fluctuate widely around it.

This example is a demonstration of how the model could be used and is not an evaluation of its accuracy.
A full benchmark study of MOMENT on anomaly detection, including the comparison with other methods,
is provided in the original paper `[1] <#references>`_.

4. Conclusion
***************************************************************************************************************

This post showed how to deploy the pre-trained MOMENT-1-large model on Amazon SageMaker
and how to use its reconstructions for zero-shot time series anomaly detection.
The model weights are used as released, with no training or fine-tuning,
and the same model serves batch transform jobs, real-time endpoints and asynchronous endpoints.
As the model artifacts are stored in Amazon S3 and nothing is downloaded at runtime, the model runs with network isolation.

The model returns the reconstructions and does not flag the anomalies.
The time steps to flag are selected by comparing the reconstruction errors with a threshold,
which still needs to be calibrated on a period of the time series that is considered normal,
for instance as a high percentile of the errors in that period.

The approach has three main practical limitations.
First, the most recent time steps are reconstructed from past observations only,
so their reconstructions are less reliable and can change when more recent data becomes available.
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

