:layout: post

.. post:: March 26, 2026
    :tags: Time Series Forecasting, Amazon Chronos, Amazon Bedrock, Strands Agents
    :category: Time Series Agents
    :author: Flavia
    :description: Build a conversational time series forecasting agent with Strands Agents using Amazon Chronos as a tool and deploy it to Amazon Bedrock AgentCore
    :keywords: Time Series Forecasting, Amazon Chronos, Amazon Bedrock AgentCore, Strands Agents
    :image_src: https://machine-learning-blog.s3.eu-west-2.amazonaws.com/forecasting_agent/architecture_diagram.png
    :image_alt: Architecture diagram showing a user communicating with a Strands agent hosted on Amazon Bedrock AgentCore Runtime, which uses Claude Sonnet 4.6 as the language model and invokes Chronos on Amazon Bedrock as a time series forecasting tool.
    :excerpt: 1

###############################################################################################################
Building a time series forecasting agent with Amazon Bedrock AgentCore
###############################################################################################################

.. raw:: html

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/forecasting_agent/architecture_diagram.png"
        style="width:100%"
        alt="Architecture diagram showing a user communicating with a Strands agent hosted on Amazon Bedrock AgentCore Runtime, which uses Claude Sonnet 4.6 as the language model and invokes Chronos on Amazon Bedrock as a time series forecasting tool."
    >

1. Overview
***************************************************************************************************************

In this post, we build a conversational time series forecasting agent.
The agent exposes a time series forecasting model as a tool, and supports multi-turn conversations
with short-term memory, allowing users to generate forecasts and iteratively adjust parameters such
as context windows, prediction horizons and quantile levels.

The forecasting tool uses Chronos `[1] <#references>`_, a family of foundation models for zero-shot
probabilistic forecasting of univariate time series trained on a large collection of real and
synthetic time series datasets using the T5 architecture `[2] <#references>`_.
Specifically, the tool uses Chronos-Bolt `[3] <#references>`_, a faster, more accurate
and more memory-efficient variant that runs on CPU and is available on Amazon Bedrock Marketplace.

The agent is built with `Strands Agents <https://strandsagents.com/>`__,
where orchestration decisions are delegated to the language model rather than defined in fixed, predefined workflows.
The agent is deployed to Amazon Bedrock AgentCore, a serverless environment designed for AI agents that provides session
isolation, short-term memory and built-in observability through Amazon CloudWatch.

2. Solution
***************************************************************************************************************

The solution consists of three steps: deploying Chronos-Bolt to Bedrock, building the Strands agent
and deploying it to Bedrock AgentCore.

2.1 Deploy Chronos to Amazon Bedrock
===============================================================================================================

We start by deploying Chronos-Bolt to a Bedrock endpoint hosted on a CPU EC2 instance
using Boto3. To use the code below, you need to provide the Bedrock Marketplace ARN of Chronos-Bolt in
your AWS region, the ARN of your Bedrock execution role, and a custom endpoint name.

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
        endpointName="<bedrock-endpoint-name>",
        acceptEula=True,
    )

    # Get the Bedrock endpoint ARN
    bedrock_endpoint_arn = response["marketplaceModelEndpoint"]["endpointArn"]

The ``bedrock_endpoint_arn`` returned by the code will be needed in two places: inside the agent's tool when
invoking the endpoint, and in a custom IAM policy that grants the AgentCore execution role permission to invoke the endpoint.

.. important::

    Remember to delete the endpoint when is no longer needed to avoid unexpected charges.

    .. code:: python

        # Delete the Bedrock endpoint
        response = bedrock_client.delete_marketplace_model_endpoint(
            endpointArn=bedrock_endpoint_arn
        )

2.2 Build the agent with Strands Agents
===============================================================================================================

To build the agent, we need three files: an empty ``__init__.py``, ``agent.py``, and ``requirements.txt``.

.. code-block:: text

   agent/
   ├── __init__.py
   ├── agent.py
   └── requirements.txt

The ``agent.py`` script implements a Strands agent backed by Claude Sonnet 4.6 on Bedrock which responds
to forecasting requests using a ``generate_forecasts`` tool. The tool takes as input the historical time series
values, the prediction length and the quantile levels, and invokes the Chronos-Bolt endpoint on Bedrock
to return the predicted mean and quantiles. The agent is wrapped in a ``BedrockAgentCoreApp`` with an async
streaming entrypoint that yields agent events - including not only text responses, but also tool call inputs
and results - back to the caller.

.. code:: python

    import json
    import boto3
    from strands import Agent, tool
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    # ── Tools ──────────────────────────────────────────────────────────────────

    # Create the Bedrock runtime client
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="<bedrock-runtime-region>"
    )


    # Define the time series forecasting tool
    @tool
    def generate_forecasts(
        target: list[float],
        prediction_length: int,
        quantile_levels: list[float]
    ) -> dict:
        """
        Generate probabilistic time series forecasts using Chronos on Amazon Bedrock.

        Parameters:
        ===============================================================================
        target: list of float.
            The historical time series values used as context.

        prediction_length: int.
            The number of future time steps to predict.

        quantile_levels: list of float.
            The quantiles to be predicted at each future time step.

        Returns:
        ===============================================================================
        dict
            Dictionary with predicted mean and quantiles at each future time step.
        """
        # Invoke the Chronos endpoint on Amazon Bedrock
        response = bedrock_runtime_client.invoke_model(
            modelId="<bedrock-endpoint-arn>",
            body=json.dumps({
                "inputs": [{
                    "target": target
                }],
                "parameters": {
                    "prediction_length": prediction_length,
                    "quantile_levels": quantile_levels
                }
            })
        )

        # Extract and return the forecasts
        forecasts = json.loads(response["body"].read()).get("predictions")[0]
        return forecasts


    # ── Agent ──────────────────────────────────────────────────────────────────

    # Create the agent
    agent = Agent(
        model="eu.anthropic.claude-sonnet-4-6",
        tools=[generate_forecasts],
        system_prompt=(
            "You are a time series forecasting assistant. "
            "When given a list of numerical values, use the `generate_forecasts` tool to produce a forecast. "
            "Always ask the user for `prediction_length` and `quantile_levels` if not provided, do not assume or default any values. "
        ),
    )

    # ── App ──────────────────────────────────────────────────────────────────

    # Create the AgentCore app
    app = BedrockAgentCoreApp()


    # Define the entrypoint of the AgentCore app
    @app.entrypoint
    async def invoke(payload: dict):
        """
        Stream agent events in response to a user message.

        Parameters:
        ===============================================================================
        payload: dict
            Request payload containing the user message under the "prompt" key.

        Yields:
        ===============================================================================
        dict
            Agent event dictionaries containing text chunks, tool use information,
            and lifecycle events emitted during agent execution.
        """
        stream = agent.stream_async(payload.get("prompt", ""))
        async for event in stream:
            yield event


    # Run the AgentCore app
    if __name__ == "__main__":
        app.run()

The ``requirements.txt`` lists the packages needed to run the agent:

.. code-block:: text

    boto3==1.42.73
    bedrock_agentcore==1.4.7
    strands-agents>=1.0.0

2.3 Deploy the agent to Amazon Bedrock AgentCore
===============================================================================================================

We deploy the agent with the `bedrock-agentcore-starter-toolkit <https://aws.github.io/bedrock-agentcore-starter-toolkit/>`__ using direct code deployment.
The ``--non-interactive`` flag skips the interactive prompts and deploys with short-term memory enabled by default.
Short-term memory persists conversation context within a session without requiring any configuration in the agent code.

.. code-block:: shell

    agentcore configure \
        --entrypoint agent.py \
        --name forecasting_agent \
        --deployment-type direct_code_deploy \
        --runtime PYTHON_3_12 \
        --requirements-file requirements.txt \
        --non-interactive

    agentcore launch

The deployment creates four resources: an AgentCore Runtime named ``forecasting_agent``,
a short-term memory resource in AgentCore Memory linked to the runtime, a deployment package uploaded as
a zip file to S3, and an AgentCore Runtime execution role in IAM.

.. important::

    Attach the following policy to the AgentCore execution role in IAM to allow the agent to invoke the
    Chronos-Bolt endpoint on Bedrock. Replace ``<bedrock-endpoint-arn>`` with the
    ``bedrock_endpoint_arn`` returned in Section 2.1.

    .. code-block:: json

        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:InvokeEndpoint"
                    ],
                    "Resource": [
                        "<bedrock-endpoint-arn>"
                    ],
                    "Condition": {
                        "StringEquals": {
                            "aws:CalledViaLast": "bedrock.amazonaws.com",
                            "aws:ResourceTag/sagemaker-sdk:bedrock": "compatible"
                        }
                    }
                }
            ]
        }

2.4 Test the agent in a Jupyter Notebook
===============================================================================================================

We start by importing the required libraries: ``boto3`` to invoke the agent on Bedrock AgentCore,
``json`` to serialize the request payload and deserialize the response, ``uuid`` to generate unique session
identifiers, and ``IPython.display`` to render the agent's responses as formatted Markdown in the notebook.

.. code:: ipython3
    # Import the required libraries
    import json
    import boto3
    import uuid
    from IPython.display import Markdown, display

Next, we set up the client and helper functions. We define the AgentCore
Runtime ARN and region, create a ``boto3`` client for invoking the
agent, and implement three helper functions:
``parse_streaming_response`` to extract agent messages from the response
stream, ``get_streaming_response`` to invoke the agent and return the
parsed messages, and ``print_messages`` to render a conversation turn in
the notebook with formatted text, tool calls and tool results.

.. code:: ipython3

    # Configure the Bedrock AgentCore runtime
    AGENTCORE_RUNTIME_ARN = "<agentcore-runtime-arn>"
    REGION = "<agentcore-runtime-region>"

    # Create the Bedrock AgentCore client
    agentcore_client = boto3.client(
        service_name="bedrock-agentcore",
        region_name=REGION
    )

    def parse_streaming_response(response: dict) -> list[dict]:
        """
        Parse the streaming response from AgentCore and extract agent messages.

        Parameters:
        ===============================================================================
        response: dict
            The raw response dictionary returned by invoke_agent_runtime,
            containing a StreamingBody object under the "response" key.

        Returns:
        ===============================================================================
        list[dict]
            List of agent message dictionaries extracted from the event stream.
        """
        messages = []
        for line in response["response"].iter_lines():
            if line:
                data = line.decode("utf-8")
                try:
                    start, end = data.index("{"), data.rindex("}")
                    event = json.loads(data[start:end + 1], strict=False)
                    if "message" in event:
                        messages.append(event["message"])
                except (ValueError, json.JSONDecodeError):
                    pass
        return messages


    def get_streaming_response(prompt: str, session_id: str) -> list[dict]:
        """
        Invoke the forecasting agent on Amazon Bedrock AgentCore and return
        the agent messages.

        Parameters:
        ===============================================================================
        prompt: str
            The user message to send to the agent.

        session_id: str
            The session identifier used to maintain conversation context across turns.

        Returns:
        ===============================================================================
        list[dict]
            List of agent message dictionaries.
        """
        # Invoke the agent on Bedrock AgentCore
        response = agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=AGENTCORE_RUNTIME_ARN,
            runtimeSessionId=session_id,
            payload=json.dumps({"prompt": prompt}),
        )

        # Parse and return the streaming response
        messages = parse_streaming_response(response)
        return messages


    def print_messages(prompt: str, messages: list[dict]) -> None:
        """
        Display a conversation turn in a Jupyter notebook, including the user
        prompt, agent text responses, tool calls and tool results.

        Parameters:
        ===============================================================================
        prompt: str
            The user message sent to the agent.

        messages: list[dict]
            List of agent message dictionaries returned by the agent.
        """
        # Display the user prompt
        display(Markdown(f"<h2>User:</h2><br>{prompt}"))

        # Display the agent response
        display(Markdown("<h2>Agent:</h2><br>"))
        for message in messages:
            for content in message["content"]:
                if "text" in content:
                    # Display text response as formatted markdown
                    display(Markdown(content["text"]))
                elif "toolUse" in content:
                    # Display tool call name and inputs
                    display(Markdown(f"🔨 Ran `{content['toolUse']['name']}`"))
                    print({"input": content["toolUse"]["input"]})
                elif "toolResult" in content:
                    # Display tool result output
                    if "content" in content["toolResult"]:
                        tool_output = content["toolResult"]["content"][0]["text"]
                        display(Markdown("🔨 Output:"))
                        print({"output": json.loads(tool_output)})

We generate a unique session ID to identify this conversation. The same
session ID will be reused across all turns to maintain conversation
context and short-term memory.

.. code:: ipython3

    # Generate the session ID
    session_id = str(uuid.uuid4())

We start the conversation by asking the agent what it can do.

.. code:: ipython3

    # Turn 1 / 3
    prompt = """
    What can you help me with?
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)

.. code-block:: text

    I'm a **time series forecasting assistant**! Here's what I can help you with:

    ---

    ### 📈 What I Do
    I generate **probabilistic time series forecasts** from your historical numerical data. This means I don't just predict a single value — I provide a **range of possible future outcomes** based on uncertainty.

    ---

    ### 🔢 What I Need From You
    To generate a forecast, I'll need:

    1. **Historical Data** — A list of numerical values representing your time series (e.g., daily sales, hourly temperatures, monthly revenue).
    2. **Prediction Length** — How many future time steps you want to forecast (e.g., 7 for the next 7 days).
    3. **Quantile Levels** — The probability levels for uncertainty bounds (e.g., `[0.1, 0.5, 0.9]` gives you a low, median, and high estimate).

    ---

    ### 💡 Example Use Cases
    - 📦 **Demand forecasting** — Predict future product sales
    - 🌡️ **Weather trends** — Forecast temperature or rainfall
    - 💰 **Financial data** — Project revenue or stock trends
    - ⚡ **Energy consumption** — Anticipate future usage
    - 🌐 **Web traffic** — Estimate future site visits

    ---

    ### 🚀 Getting Started
    Simply share your **time series data**, and let me know:
    - How far ahead you'd like to forecast
    - What quantile levels you want

    I'll take care of the rest!

We then provide a time series and ask the agent to forecast the next 10 values. The agent
asks for the quantile levels before proceeding, as instructed by the system prompt.

.. code:: ipython3

    # Turn 2 / 3
    prompt = """
    Can we predict the next 10 values of this time series?
    [16, 6, 0, 3, 13, 20, 18, 9, 1, 1, 10, 18, 19, 12, 3, 0, 7, 16, 20, 15, 5, 0, 4, 14, 20, 17, 8, 1, 2, 11, 19, 19, 11, 2, 1, 8, 17, 20, 14, 4, 0, 5, 14, 20, 16, 7, 0, 3, 12, 19, 18, 10, 2, 1, 9, 18, 20, 13, 3, 0, 6, 15, 20, 16]
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)

.. code-block:: text

    Sure! Before I generate the forecast, I just need one more detail:

    **What quantile levels would you like?**
    These represent the uncertainty bounds of the forecast. For example:
    - `[0.1, 0.5, 0.9]` → low, median, and high estimates
    - `[0.25, 0.5, 0.75]` → tighter range around the median

    What would you prefer?

We provide the requested quantile levels and specify that we want the mean instead of
the median. The agent now has all the information it needs and calls the ``generate_forecasts`` tool, returning the mean
forecast and the 95% prediction interval. This turn demonstrates the short-term memory capability - the agent recalls
the time series and prediction length from the previous turn without us repeating them.

.. code:: ipython3

    # Turn 3 / 3
    prompt = """
    I need a 95% prediction interval. I don't need the median, only the mean.
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)

.. code-block:: text

    Got it! I'll use quantile levels `[0.025, 0.975]` for the 95% prediction interval, along with the mean.

    🔨 Ran `generate_forecasts`

        {'input': {'target': [16, 6, 0, 3, 13, 20, 18, 9, 1, 1, 10, 18, 19, 12, 3, 0, 7, 16, 20, 15, 5, 0, 4, 14, 20, 17, 8, 1, 2, 11, 19, 19, 11, 2, 1, 8, 17, 20, 14, 4, 0, 5, 14, 20, 16, 7, 0, 3, 12, 19, 18, 10, 2, 1, 9, 18, 20, 13, 3, 0, 6, 15, 20, 16], 'prediction_length': 10, 'quantile_levels': [0.025, 0.975]}}

    🔨 Output:

        {'output': {'mean': [7.842952728271484, 3.9406585693359375, 4.136490345001221, 7.826767921447754, 15.442928314208984, 15.398470878601074, 9.315390586853027, 5.722835063934326, 5.371423244476318, 7.833820343017578], '0.025': [2.4480767250061035, -0.8402576446533203, -0.2643604278564453, 2.227060317993164, 7.496842384338379, 7.033439636230469, 2.4833216667175293, -0.3372507095336914, -0.5436820983886719, 0.8084659576416016], '0.975': [13.956232070922852, 10.066437721252441, 10.064048767089844, 14.887125015258789, 23.994970321655273, 24.602264404296875, 17.92835235595703, 13.450220108032227, 13.074174880981445, 16.784805297851562]}}

    Here are the forecast results for the next 10 time steps:

    | Step | Mean | Lower 2.5% | Upper 97.5% |
    |------|------|------------|-------------|
    | 1  | 7.84  | 2.45  | 13.96 |
    | 2  | 3.94  | -0.84 | 10.07 |
    | 3  | 4.14  | -0.26 | 10.06 |
    | 4  | 7.83  | 2.23  | 14.89 |
    | 5  | 15.44 | 7.50  | 23.99 |
    | 6  | 15.40 | 7.03  | 24.60 |
    | 7  | 9.32  | 2.48  | 17.93 |
    | 8  | 5.72  | -0.34 | 13.45 |
    | 9  | 5.37  | -0.54 | 13.07 |
    | 10 | 7.83  | 0.81  | 16.78 |

You can download the full code from our `GitHub repository <https://github.com/flaviagiammarino/machine-learning-blog/tree/main/forecasting_agent/>`__.

References
***************************************************************************************************************

[1] Ansari, A.F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S.S., Arango, S.P., Kapoor, S. and Zschiegner, J., (2024).
Chronos: Learning the language of time series. *arXiv preprint*,
`doi: 10.48550/arXiv.2403.07815 <https://doi.org/10.48550/arXiv.2403.07815>`__.

[2] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu, P.J., (2020).
Exploring the limits of transfer learning with a unified text-to-text transformer.
*Journal of machine learning research*, 21(140), pp.1-67,
`doi: 10.48550/arXiv.1910.10683 <https://doi.org/10.48550/arXiv.1910.10683>`__.

[3] Ansari, A.F., Turkmen, C., Shchur, O., and Stella, L. (2024).
Fast and accurate zero-shot forecasting with Chronos-Bolt and AutoGluon.
`AWS Blogs - Artificial Intelligence <https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/>`__.
