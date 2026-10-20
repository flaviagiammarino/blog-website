:layout: post

.. post:: October 16, 2026
    :tags: Amazon Bedrock, Amazon SageMaker, Strands Agents, ClickHouse
    :category: Agentic AI
    :author: Flavia
    :description: Build and deploy a ClickHouse text-to-SQL agent with Amazon SageMaker, Amazon Bedrock AgentCore and Strands Agents
    :keywords: Text-to-SQL, ClickHouse, Amazon SageMaker, Amazon Bedrock, Strands Agents
    :image_src: https://machine-learning-blog.s3.eu-west-2.amazonaws.com/clickhouse_agent/architecture_diagram.png
    :image_alt: Architecture diagram showing a user invoking a Strands agent on Amazon Bedrock AgentCore Runtime in a private VPC subnet, where the agent calls an open-weight LLM on a private Amazon SageMaker endpoint and queries a ClickHouse database through an AWS PrivateLink endpoint.
    :summary: A Strands agent deployed to Amazon Bedrock AgentCore Runtime that translates natural language questions into SQL queries against a ClickHouse database using an open-weight LLM served from a private Amazon SageMaker endpoint. The agent runs in a private VPC subnet, reaches the model and the database over private endpoints, and communicates with AWS services through a NAT gateway.

########################################################################################################################
Running a self-hosted open-weight LLM agent on ClickHouse data with Amazon SageMaker and Amazon Bedrock AgentCore
########################################################################################################################

.. raw:: html

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/clickhouse_agent/architecture_diagram.png"
        style="width:100%"
        alt="Architecture diagram showing a user invoking a Strands agent on Amazon Bedrock AgentCore Runtime in a private VPC subnet, where the agent calls an open-weight LLM on a private Amazon SageMaker endpoint and queries a ClickHouse database through an AWS PrivateLink endpoint."
    >

1. Overview
***************************************************************************************************************

Commercial model APIs provide the simplest path to integrating LLMs into applications, but they do not meet every organization's requirements.
Data sovereignty rules can prevent internal data from being sent to a third-party provider, which is a particular constraint for agents that work on the organization's own data.

A text-to-SQL agent, for instance, sends the database schema, the user questions and the query results to the model at every step,
so any external model provider would receive a continuous stream of internal data.
For organizations subject to such rules, the model has to run on infrastructure they control,
and the network path between the agent, the model and the database has to stay private.

In this post, we build a text-to-SQL agent using an open-weight LLM.
The model, Qwen3.6-27B, is deployed to a private Amazon SageMaker endpoint built on the AWS Deep Learning Container for vLLM,
with network isolation enabled.
Its weights are stored in an S3 bucket and loaded from there over an S3 gateway endpoint, instead of being downloaded from the Hugging Face Hub.

The agent is built with Strands Agents and deployed to an Amazon Bedrock AgentCore Runtime in the same private subnet as the SageMaker endpoint.
It queries the database via the ClickHouse MCP server over an AWS PrivateLink interface endpoint in the same subnet,
with the database's public endpoint disabled.
The subnet has no inbound path from the internet, and its outbound traffic to AWS services goes through a NAT gateway.

2. Solution
***************************************************************************************************************

The solution involves setting up the VPC and private connectivity, uploading the model weights to S3 and deploying the model to a SageMaker endpoint, building the agent in Python with Strands Agents, containerizing it with Docker, pushing the image to Amazon ECR, and deploying it to a Bedrock AgentCore Runtime. We deploy all resources in the AWS region where the ClickHouse Cloud service is hosted.
The code is organized as follows:

.. code-block:: text

    .
    ├── model.py                  # SageMaker model, endpoint configuration and endpoint
    └── agent/
        ├── agent.py              # Strands agent with the SageMaker model and ClickHouse tools
        ├── Dockerfile            # Container image definition
        └── build_and_deploy.sh   # ECR image creation and AgentCore Runtime deployment

To follow along, you will need an AWS account with sufficient IAM permissions to create and manage the resources described in this post, a ClickHouse Cloud service on the Scale or Enterprise plan (required for PrivateLink), the AWS CLI installed and configured with appropriate credentials, the Hugging Face CLI for the initial download of the model weights, Docker for building and pushing the container image, and Python 3.13 with ``uv`` for project setup and dependency management.

2.1 Set up the VPC and private connectivity
===============================================================================================================

We create a private subnet with its own route table, place a regional NAT gateway in the VPC, and route the subnet's outbound traffic through it. This gives the workloads in the subnet outbound access, which they use to reach AWS services, while leaving them with no public address and no inbound path. We also associate an S3 gateway endpoint with the subnet's route table, so that S3 traffic bypasses the NAT gateway. The endpoint policy restricts access to two buckets: the one holding the model weights, and the regional bucket in which Amazon ECR stores container image layers.

The SageMaker endpoint attaches its network interfaces to the private subnet through the model's VPC configuration. Network isolation is enabled on the model, so the container makes no outbound calls; SageMaker downloads the weights on its behalf over the S3 gateway endpoint.

The ClickHouse database is reached through PrivateLink. We create an interface endpoint to ClickHouse's regional endpoint service in the private subnet and register it with the service in the ClickHouse Cloud console. Private DNS is enabled on the endpoint, so inside the VPC the service's private hostname resolves to the endpoint's private IP address, and database traffic goes through PrivateLink instead of the NAT gateway. We empty the database's IP access list, which closes the public hostname and leaves the private endpoint as the only path to the data.

The agent is deployed on AgentCore Runtime in VPC network mode, in the private subnet.

Each component has its own security group:

* ``<model-security-group-id>``, attached to the SageMaker endpoint: no inbound rules; outbound HTTPS to the S3 prefix list only, for the weights download.
* ``<agent-security-group-id>``, attached to the AgentCore Runtime: no inbound rules; outbound HTTPS to any destination, for the agent's calls to AWS services through the NAT gateway, and port 8443 to ``<clickhouse-security-group-id>``, for its database connections.
* ``<clickhouse-security-group-id>``, attached to the ClickHouse interface endpoint: inbound on ClickHouse's TLS ports, 8443 and 9440, from ``<agent-security-group-id>`` only; no outbound rules.

Security groups are stateful, so the responses to connections the agent opens are allowed back without inbound rules.

2.2 Deploy the open-weight LLM to SageMaker
===============================================================================================================

We create the SageMaker model with the vLLM container, pointing it to the uncompressed model weights in S3, and attach it to the private subnet and the model's security group created in Section 2.1, with network isolation enabled.

We configure the vLLM server with a maximum context length of 131,072 tokens and a GPU memory utilization of 90%, enable automatic tool choice with the ``qwen3_xml`` tool call parser, enable the ``qwen3`` reasoning parser to separate the model's reasoning from its final answer, and shard the model across the four GPUs of the ``ml.g6.12xlarge`` instance with tensor parallelism.

In the endpoint configuration, we deploy a single instance and increase the model download and container startup health check timeouts to 30 minutes, to allow enough time for the weights to be downloaded and loaded onto the GPUs. Finally, we create the endpoint from the endpoint configuration.

.. code:: python

    import boto3

    # AWS region for all resources
    region = "<aws-region>"

    # S3 prefix holding the model weights
    model_uri = "<s3-uri>"

    # IAM execution role for the SageMaker endpoint
    execution_role_arn = "<sagemaker-execution-role-arn>"

    # Private subnet
    subnets = ["<private-subnet-id>"]

    # Security groups
    security_groups = ["<model-security-group-id>"]

    # vLLM inference container from the AWS Deep Learning Containers registry
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/vllm:server-sagemaker-cuda-v2.5"

    # SageMaker model, endpoint configuration, and endpoint names
    model_name = "qwen36-27b-fp8"
    endpoint_config_name = "qwen36-27b-fp8-cfg"
    endpoint_name = "qwen36-27b-fp8"

    # SageMaker instance type
    instance_type = "ml.g6.12xlarge"

    # Create a SageMaker client
    sm = boto3.client("sagemaker", region_name=region)

    # Create the model
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=execution_role_arn,
        EnableNetworkIsolation=True,
        VpcConfig={
            "Subnets": subnets,
            "SecurityGroupIds": security_groups
        },
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataSource": {
                "S3DataSource": {
                    "S3Uri": model_uri,
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None"
                }
            },
            "Environment": {
                "SM_VLLM_SERVED_MODEL_NAME": "qwen3.6-27b",
                "SM_VLLM_MAX_MODEL_LEN": "131072",
                "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.90",
                "SM_VLLM_ENABLE_AUTO_TOOL_CHOICE": "true",
                "SM_VLLM_TOOL_CALL_PARSER": "qwen3_xml",
                "SM_VLLM_REASONING_PARSER": "qwen3",
                "SM_VLLM_TENSOR_PARALLEL_SIZE": "4"
            },
        },
    )

    # Create the endpoint configuration
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
            "ModelDataDownloadTimeoutInSeconds": 1800,
            "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
        }],
    )

    # Create the endpoint from the configuration
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )


2.3 Build the agent with Strands Agents
===============================================================================================================

The agent uses the Strands ``SageMakerAIModel`` provider to call the private SageMaker endpoint, with streaming enabled and a maximum of 16,384 output tokens.

The agent connects to ClickHouse with the `mcp-clickhouse <https://github.com/ClickHouse/mcp-clickhouse>`__ server, which includes tools for listing databases and tables and running SQL queries. The agent retrieves the database host, port, user and password from AWS Secrets Manager, while the database name is passed as an environment variable to the AgentCore Runtime by the deployment script in the next section.

The host stored in the secret is the service's private hostname, of the form ``<id>.<region>.vpce.aws.clickhouse.cloud``, which resolves to the PrivateLink endpoint inside the VPC, shown as the DNS name in the ClickHouse Cloud's private endpoint settings. With the public hostname, the connection would leave the subnet through the NAT gateway and be refused, as the database's IP access list is empty.

.. code:: python

    import os
    import json
    import boto3
    from strands import Agent
    from strands.tools.mcp import MCPClient
    from strands.models.sagemaker import SageMakerAIModel
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    # Create a Secrets Manager client and retrieve the database credentials
    client = boto3.client("secretsmanager")
    secret = client.get_secret_value(SecretId=os.environ["SECRET_ID"])

    # Parse the secret string and extract the connection details
    secret_string = json.loads(secret["SecretString"])
    db_config = {
        "host": secret_string["private_host"],
        "port": str(secret_string["port"]),
        "user": secret_string["user"],
        "password": secret_string["password"],
        "name": os.environ["DB_NAME"],
    }

    def create_clickhouse_client():
        # Configure the mcp-clickhouse server to run as a subprocess
        server_params = StdioServerParameters(
            command="mcp-clickhouse",
            env={
                "CLICKHOUSE_HOST": db_config["host"],
                "CLICKHOUSE_PORT": db_config["port"],
                "CLICKHOUSE_USER": db_config["user"],
                "CLICKHOUSE_PASSWORD": db_config["password"],
                "CLICKHOUSE_DATABASE": db_config["name"],
                "CLICKHOUSE_SECURE": "true",
                "CLICKHOUSE_CONNECT_TIMEOUT": "10",
                "CLICKHOUSE_SEND_RECEIVE_TIMEOUT": "60",
            }
        )
        # Return an MCP client that communicates with the server over stdio
        return MCPClient(lambda: stdio_client(server_params))

    # Start the MCP client and keep it alive for the lifetime of the session
    clickhouse_client = create_clickhouse_client()
    clickhouse_client.__enter__()

    # Retrieve the list of tools exposed by the mcp-clickhouse server
    tools = clickhouse_client.list_tools_sync()

    # Configure the model served by the private SageMaker endpoint
    model = SageMakerAIModel(
        endpoint_config={
            "endpoint_name": os.environ["ENDPOINT_NAME"],
            "region_name": os.environ["AWS_REGION"],
        },
        payload_config={
            "max_tokens": 16384,
            "stream": True,
        }
    )

    # Initialize the Strands agent with the SageMaker model and the ClickHouse tools
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a data analyst assistant that answers questions about a ClickHouse database. "
            "Before writing any query, inspect the schema with the available tools to find the relevant tables and columns. "
            "If a query fails, read the error message, correct the query, and retry once. "
            "In your answer, state the result and include the final query you ran. "
            "You have read-only access; decline requests to modify data. "
            "If the available data cannot answer the question, say so."
        )
    )

    # Create the AgentCore app
    app = BedrockAgentCoreApp()

    # Define the async streaming entrypoint
    @app.entrypoint
    async def invoke(payload: dict):
        # Stream agent events including text chunks, tool use, and results
        stream = agent.stream_async(payload.get("prompt", ""))
        async for event in stream:
            yield event

    # Run the AgentCore app
    if __name__ == "__main__":
        app.run()

The ``Dockerfile`` installs the dependencies without development packages and adds the virtual environment to the ``PATH``, so that the ``mcp-clickhouse`` executable can be launched directly as a subprocess by the agent.

.. code:: docker

    # Use uv's ARM64 Python 3.13 base image
    FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.13-bookworm-slim

    # Set the working directory inside the container
    WORKDIR /app

    # Copy uv files
    COPY pyproject.toml uv.lock ./

    # Install dependencies
    RUN uv sync --frozen --no-cache --no-dev

    # Put virtual environment on PATH
    ENV PATH="/app/.venv/bin:$PATH"

    # Copy Python modules
    COPY agent.py ./

    # Expose port
    EXPOSE 8080

    # Run application
    CMD ["python", "agent.py"]

2.4 Deploy the agent to Amazon Bedrock AgentCore
===============================================================================================================

The deployment script initializes a ``uv`` project and adds the required dependencies, including the ``sagemaker`` extra of Strands Agents for the SageMaker model provider. This generates the ``pyproject.toml`` and ``uv.lock`` files used by the Dockerfile.

It then authenticates with ECR, creates a repository, and builds and pushes the ARM64 container image. Finally, it deploys the agent to AgentCore Runtime in VPC mode, specifying the private subnet and the agent's security group created in Section 2.1. The secret ID, the database name and the SageMaker endpoint name are passed as environment variables and retrieved at runtime.

.. code:: bash

    # AWS account ID
    aws_account_id="<aws-account-id>"

    # AWS region for all resources
    region="<aws-region>"

    # AgentCore Runtime name
    agent_name="<agentcore-runtime-name>"

    # ECR repository name for the container image
    repository_name="<ecr-repository-name>"

    # Secrets Manager secret ID for ClickHouse credentials
    secret_id="<secret-name>"

    # ClickHouse database name
    db_name="<clickhouse-db-name>"

    # SageMaker endpoint name
    endpoint_name="<sagemaker-endpoint-name>"

    # IAM execution role for AgentCore Runtime
    role_name="<agentcore-runtime-role>"

    # Private subnet
    subnet_id="<private-subnet-id>"

    # Security group
    security_group_id="<agent-security-group-id>"

    # Project setup
    uv init --name "$agent_name" --description "Strands Agent for ClickHouse Text-to-SQL" --python 3.13 --bare
    uv add "strands-agents[sagemaker]" mcp-clickhouse boto3 bedrock-agentcore

    # ECR login and repository creation
    aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$region.amazonaws.com
    aws ecr create-repository --repository-name $repository_name --region $region

    # Build and push container image
    docker buildx build --platform linux/arm64 -t $aws_account_id.dkr.ecr.$region.amazonaws.com/$repository_name:latest --push .

    # Deploy AgentCore Runtime
    aws bedrock-agentcore-control create-agent-runtime \
      --agent-runtime-name "$agent_name" \
      --agent-runtime-artifact "{
        \"containerConfiguration\": {
          \"containerUri\": \"${aws_account_id}.dkr.ecr.${region}.amazonaws.com/${repository_name}:latest\"
        }
      }" \
      --environment-variables "SECRET_ID=${secret_id},DB_NAME=${db_name},ENDPOINT_NAME=${endpoint_name}" \
      --role-arn "arn:aws:iam::${aws_account_id}:role/${role_name}" \
      --region "$region" \
      --network-configuration "{
        \"networkMode\": \"VPC\",
        \"networkModeConfig\": {
          \"subnets\": [\"${subnet_id}\"],
          \"securityGroups\": [\"${security_group_id}\"]
        }
      }"


2.5 Test the agent in a Jupyter Notebook
===============================================================================================================

After the agent has been deployed, we test it in a Jupyter Notebook on a small, synthetically generated retail banking dataset.

The database contains three tables: ``customers``, with 1,000 customers, ``accounts``, with one current, savings or credit card account per customer, and ``transactions``, with 10,000 transactions over the last 12 months, where debits are recorded as negative amounts and salary payments as positive ones.

We start by importing the required libraries: ``boto3`` to invoke the agent on Bedrock AgentCore,
``json`` to serialize the request payload and deserialize the response, ``uuid`` to generate unique session
identifiers, and ``IPython.display`` to render the agent's responses as formatted Markdown in the notebook.

.. code:: ipython3

    # Import the required libraries
    import json
    import boto3
    import uuid
    from botocore.config import Config
    from IPython.display import Markdown, display

Next, we set up the client and helper functions. We define the AgentCore Runtime ARN and region,
create a ``boto3`` client for invoking the agent, and implement three helper functions:
``parse_streaming_response`` to extract agent messages from the response stream,
``get_streaming_response`` to invoke the agent and return the parsed messages,
and ``print_messages`` to render a conversation turn in the notebook with formatted text,
tool calls and tool results.

.. code:: ipython3

    # Configure the Bedrock AgentCore runtime
    AGENTCORE_RUNTIME_ARN = "<agentcore-runtime-arn>"
    REGION = "<agentcore-runtime-region>"

    # Create the Bedrock AgentCore client
    agentcore_client = boto3.client(
        service_name="bedrock-agentcore",
        region_name=REGION,
        config=Config(
            read_timeout=900,
            connect_timeout=10,
            retries={"max_attempts": 0},
        ),
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
        for raw in response["response"].iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if line.startswith("data: "):
                line = line[len("data: "):]
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if "message" in event:
                messages.append(event["message"])
            elif "error" in event:
                raise RuntimeError(f"Agent error: {event['error']}")
        return messages


    def get_streaming_response(prompt: str, session_id: str) -> list[dict]:
        """
        Invoke the agent on Amazon Bedrock AgentCore and return the agent messages.

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
                    display(Markdown("🔨 Output:"))
                    for block in content["toolResult"].get("content", []):
                        print({"output": block.get("text", block.get("json"))})

We generate a unique session ID to identify this conversation. The same
session ID will be reused across all turns to maintain conversation
context and short-term memory.

.. code:: ipython3

    # Generate the session ID
    session_id = str(uuid.uuid4())

We start the conversation by asking the agent what it can do.

.. code:: ipython3

    # Turn 1 / 4
    prompt = """
    What can you help me with?
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)

.. raw:: html

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    What can you help me with?
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Agent</div>
    <div class="message">
    <p>I can help you explore and analyze data in a ClickHouse database! Here's what I can do:</p>
    <h2>📊 Database Exploration</h2>
    <ul>
    <li><strong>List available databases</strong> - See what databases exist in the system</li>
    <li><strong>List tables in a database</strong> - Discover tables with their schemas, row counts, and column information</li>
    <li><strong>Inspect table structures</strong> - Understand column names, data types, and table metadata</li>
    </ul>
    <h2>🔍 Data Analysis</h2>
    <ul>
    <li><strong>Run SQL queries</strong> - Execute read-only SELECT queries to extract and analyze data</li>
    <li><strong>Data exploration</strong> - Help you understand patterns, aggregations, and insights in your data</li>
    <li><strong>Complex queries</strong> - Build queries with JOINs, GROUP BY, window functions, and more</li>
    </ul>
    <h2>🛠️ Query Support</h2>
    <ul>
    <li><strong>Schema validation</strong> - Check if a query will work by inspecting output columns and types</li>
    <li><strong>Performance estimation</strong> - Preview query complexity before execution</li>
    <li><strong>Error troubleshooting</strong> - Fix and retry queries that fail</li>
    </ul>
    <h2>⚠️ Limitations</h2>
    <ul>
    <li><strong>Read-only access</strong> - I can only run SELECT queries; no data modification (INSERT, UPDATE, DELETE, DROP)</li>
    <li><strong>ClickHouse-specific</strong> - I work with ClickHouse SQL syntax and features</li>
    </ul>
    <h2>🚀 Getting Started</h2>
    <p>To help you, I can:</p>
    <ol>
    <li>Show you what databases are available</li>
    <li>List tables in a specific database</li>
    <li>Run analytical queries based on your questions</li>
    <li>Explain the results</li>
    </ol>
    <p>Would you like me to start by showing you what databases are available, or do you have a specific question about the data?</p>
    </div>
    </div>

We then ask the agent to list the tables in the database.

.. code:: ipython3

    # Turn 2 / 4
    prompt = """
    What tables are in the database?
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)


.. raw:: html

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    What tables are in the database?
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Agent</div>
    <div class="message">
    <p>🔨 Ran <code>list_databases</code></p>
    <pre class="tool-x-scroll">
    {'input': {}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">
    {'output': '["INFORMATION_SCHEMA", "default", "information_schema", "system"]'}
    </pre>
    <p>🔨 Ran <code>list_tables</code></p>
    <pre class="tool-x-scroll">
    {'input': {'database': 'default'}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">{'output': '{"tables": [{"database": "default", "name": "accounts", "engine": "SharedMergeTree", "create_table_query": "CREATE TABLE default.accounts (`account_id` UInt32, `customer_id` UInt32, `account_type` Enum8(\'current\' = 1, \'savings\' = 2, \'credit_card\' = 3), `opened_date` Date, `status` Enum8(\'active\' = 1, \'frozen\' = 2, \'closed\' = 3)) ENGINE = SharedMergeTree(\'/clickhouse/tables/{uuid}/{shard}\', \'{replica}\') ORDER BY (customer_id, account_id) SETTINGS index_granularity = 8192", "dependencies_database": [], "dependencies_table": [], "engine_full": "SharedMergeTree(\'/clickhouse/tables/{uuid}/{shard}\', \'{replica}\') ORDER BY (customer_id, account_id) SETTINGS index_granularity = 8192", "sorting_key": "customer_id, account_id", "primary_key": "customer_id, account_id", "total_rows": 1000, "total_bytes": 8381, "total_bytes_uncompressed": 12192, "parts": 1, "active_parts": 1, "total_marks": 2, "comment": "", "columns": [{"database": "default", "table": "accounts", "name": "account_id", "column_type": "UInt32", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "accounts", "name": "customer_id", "column_type": "UInt32", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "accounts", "name": "account_type", "column_type": "Enum8(\'current\' = 1, \'savings\' = 2, \'credit_card\' = 3)", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "accounts", "name": "opened_date", "column_type": "Date", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "accounts", "name": "status", "column_type": "Enum8(\'active\' = 1, \'frozen\' = 2, \'closed\' = 3)", "default_kind": "", "default_expression": "", "comment": ""}]}, {"database": "default", "name": "customers", "engine": "SharedMergeTree", "create_table_query": "CREATE TABLE default.customers (`customer_id` UInt32, `first_name` String, `last_name` String, `date_of_birth` Date32, `customer_since` Date) ENGINE = SharedMergeTree(\'/clickhouse/tables/{uuid}/{shard}\', \'{replica}\') ORDER BY customer_id SETTINGS index_granularity = 8192", "dependencies_database": [], "dependencies_table": [], "engine_full": "SharedMergeTree(\'/clickhouse/tables/{uuid}/{shard}\', \'{replica}\') ORDER BY customer_id SETTINGS index_granularity = 8192", "sorting_key": "customer_id", "primary_key": "customer_id", "total_rows": 1000, "total_bytes": 13112, "total_bytes_uncompressed": 24274, "parts": 1, "active_parts": 1, "total_marks": 2, "comment": "", "columns": [{"database": "default", "table": "customers", "name": "customer_id", "column_type": "UInt32", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "customers", "name": "first_name", "column_type": "String", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "customers", "name": "last_name", "column_type": "String", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "customers", "name": "date_of_birth", "column_type": "Date32", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "customers", "name": "customer_since", "column_type": "Date", "default_kind": "", "default_expression": "", "comment": ""}]}, {"database": "default", "name": "transactions", "engine": "SharedMergeTree", "create_table_query": "CREATE TABLE default.transactions (`transaction_id` UInt32, `account_id` UInt32, `transaction_date` Date, `amount` Decimal(12, 2), `category` LowCardinality(String)) ENGINE = SharedMergeTree(\'/clickhouse/tables/{uuid}/{shard}\', \'{replica}\') ORDER BY (account_id, transaction_date) SETTINGS index_granularity = 8192", "dependencies_database": [], "dependencies_table": [], "engine_full": "SharedMergeTree(\'/clickhouse/tables/{uuid}/{shard}\', \'{replica}\') ORDER BY (account_id, transaction_date) SETTINGS index_granularity = 8192", "sorting_key": "account_id, transaction_date", "primary_key": "account_id, transaction_date", "total_rows": 10000, "total_bytes": 67969, "total_bytes_uncompressed": 190357, "parts": 1, "active_parts": 1, "total_marks": 2, "comment": "", "columns": [{"database": "default", "table": "transactions", "name": "transaction_id", "column_type": "UInt32", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "transactions", "name": "account_id", "column_type": "UInt32", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "transactions", "name": "transaction_date", "column_type": "Date", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "transactions", "name": "amount", "column_type": "Decimal(12, 2)", "default_kind": "", "default_expression": "", "comment": ""}, {"database": "default", "table": "transactions", "name": "category", "column_type": "LowCardinality(String)", "default_kind": "", "default_expression": "", "comment": ""}]}], "next_page_token": null, "total_tables": 3}'}
    </pre>
    <p>Here are the tables in the <strong>default</strong> database:</p>
    <h2>📋 Available Tables</h2>
    <h3>1. <code>accounts</code> (1,000 rows)</h3>
    <table>
    <thead>
    <tr>
    <th>Column</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><code>account_id</code></td>
    <td>UInt32</td>
    <td>Unique account identifier</td>
    </tr>
    <tr>
    <td><code>customer_id</code></td>
    <td>UInt32</td>
    <td>References customer</td>
    </tr>
    <tr>
    <td><code>account_type</code></td>
    <td>Enum8</td>
    <td>'current', 'savings', 'credit_card'</td>
    </tr>
    <tr>
    <td><code>opened_date</code></td>
    <td>Date</td>
    <td>When the account was opened</td>
    </tr>
    <tr>
    <td><code>status</code></td>
    <td>Enum8</td>
    <td>'active', 'frozen', 'closed'</td>
    </tr>
    </tbody>
    </table>
    <h3>2. <code>customers</code> (1,000 rows)</h3><table>
    <thead>
    <tr>
    <th>Column</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><code>customer_id</code></td>
    <td>UInt32</td>
    <td>Unique customer identifier</td>
    </tr>
    <tr>
    <td><code>first_name</code></td>
    <td>String</td>
    <td>Customer's first name</td>
    </tr>
    <tr>
    <td><code>last_name</code></td>
    <td>String</td>
    <td>Customer's last name</td>
    </tr>
    <tr>
    <td><code>date_of_birth</code></td>
    <td>Date32</td>
    <td>Customer's date of birth</td>
    </tr>
    <tr>
    <td><code>customer_since</code></td>
    <td>Date</td>
    <td>When the customer started</td>
    </tr>
    </tbody>
    </table>
    <h3>3. <code>transactions</code> (10,000 rows)</h3>
    <table>
    <thead>
    <tr>
    <th>Column</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><code>transaction_id</code></td>
    <td>UInt32</td>
    <td>Unique transaction identifier</td>
    </tr>
    <tr>
    <td><code>account_id</code></td>
    <td>UInt32</td>
    <td>References account</td>
    </tr>
    <tr>
    <td><code>transaction_date</code></td>
    <td>Date</td>
    <td>Date of the transaction</td>
    </tr>
    <tr>
    <td><code>amount</code></td>
    <td>Decimal(12, 2)</td>
    <td>Transaction amount</td>
    </tr>
    <tr>
    <td><code>category</code></td>
    <td>LowCardinality(String)</td>
    <td>Transaction category</td>
    </tr>
    </tbody>
    </table>
    <hr/>
    <p>This appears to be a <strong>financial/banking dataset</strong> with customers, their accounts, and transaction history.</p>
    <p>What would you like to analyze? For example:</p>
    <ul>
    <li>Customer demographics and account distribution</li>
    <li>Transaction patterns and spending trends</li>
    <li>Account status analysis</li>
    <li>Revenue/flow analysis by category or time period</li>
    </ul>
    </div>
    </div>

Next, we ask the agent a question that relies on context from the previous turn to demonstrate short-term memory capabilities.

.. code:: ipython3

    # Turn 3 / 4
    prompt = """
    Describe the last one and show me the first five rows.
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)


.. raw:: html

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Describe the last one and show me the first five rows.
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Agent</div>
    <div class="message">
    <p>The last table is <strong><code>transactions</code></strong>, which contains financial transaction records.</p>
    <h2>📊 Table Description: <code>transactions</code></h2><ul>
    <li><strong>Rows</strong>: 10,000</li>
    <li><strong>Engine</strong>: SharedMergeTree</li>
    <li><strong>Primary Key</strong>: <code>account_id</code>, <code>transaction_date</code></li>
    </ul>
    <table>
    <thead>
    <tr>
    <th>Column</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><code>transaction_id</code></td>
    <td>UInt32</td>
    <td>Unique transaction identifier</td>
    </tr>
    <tr>
    <td><code>account_id</code></td>
    <td>UInt32</td>
    <td>References the account this transaction belongs to</td>
    </tr>
    <tr>
    <td><code>transaction_date</code></td>
    <td>Date</td>
    <td>Date when the transaction occurred</td>
    </tr>
    <tr>
    <td><code>amount</code></td>
    <td>Decimal(12, 2)</td>
    <td>Transaction amount (can be positive or negative)</td>
    </tr>
    <tr>
    <td><code>category</code></td>
    <td>LowCardinality(String)</td>
    <td>Transaction category (e.g., food, transport, etc.)</td>
    </tr>
    </tbody>
    </table>
    <p>Now let me show you the first 5 rows:</p>
    <p>🔨 Ran <code>run_query</code></p>
    <pre>{'input': {'query': 'SELECT * FROM default.transactions LIMIT 5'}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">{'output': '{"columns": ["transaction_id", "account_id", "transaction_date", "amount", "category"], "rows": [[5940, 10110974, "2025-10-13", "-21.68", "shopping"], [9980, 10110974, "2026-01-18", "-53.98", "restaurants"], [7065, 10110974, "2026-03-29", "-126.91", "transfer"], [1668, 10110974, "2026-04-09", "-22.96", "transport"], [6020, 10110974, "2026-05-01", "-43.00", "restaurants"]]}'}
    </pre>
    <h2>📊 Transactions Table Description</h2>
    <p>The <strong><code>transactions</code></strong> table contains financial transaction records with the following structure:</p>
    <table>
    <thead>
    <tr>
    <th>Column</th>
    <th>Type</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><code>transaction_id</code></td>
    <td>UInt32</td>
    <td>Unique identifier for each transaction</td>
    </tr>
    <tr>
    <td><code>account_id</code></td>
    <td>UInt32</td>
    <td>References the account this transaction belongs to</td>
    </tr>
    <tr>
    <td><code>transaction_date</code></td>
    <td>Date</td>
    <td>Date when the transaction occurred</td>
    </tr>
    <tr>
    <td><code>amount</code></td>
    <td>Decimal(12, 2)</td>
    <td>Transaction amount (negative values indicate expenses/outflow)</td>
    </tr>
    <tr>
    <td><code>category</code></td>
    <td>LowCardinality(String)</td>
    <td>Category of the transaction (e.g., shopping, restaurants, etc.)</td>
    </tr>
    </tbody>
    </table>
    <p><strong>Table Stats:</strong></p>
    <ul>
    <li><strong>Rows:</strong> 10,000</li>
    <li><strong>Engine:</strong> SharedMergeTree</li>
    <li><strong>Sorted by:</strong> <code>account_id, transaction_date</code></li>
    </ul>
    <hr/>
    <h2 id="%F0%9F%93%8B-First-5-Rows">📋 First 5 Rows<a class="anchor-link" href="#%F0%9F%93%8B-First-5-Rows">¶</a></h2><table>
    <thead>
    <tr>
    <th>transaction_id</th>
    <th>account_id</th>
    <th>transaction_date</th>
    <th>amount</th>
    <th>category</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>5940</td>
    <td>10110974</td>
    <td>2025-10-13</td>
    <td>-21.68</td>
    <td>shopping</td>
    </tr>
    <tr>
    <td>9980</td>
    <td>10110974</td>
    <td>2026-01-18</td>
    <td>-53.98</td>
    <td>restaurants</td>
    </tr>
    <tr>
    <td>7065</td>
    <td>10110974</td>
    <td>2026-03-29</td>
    <td>-126.91</td>
    <td>transfer</td>
    </tr>
    <tr>
    <td>1668</td>
    <td>10110974</td>
    <td>2026-04-09</td>
    <td>-22.96</td>
    <td>transport</td>
    </tr>
    <tr>
    <td>6020</td>
    <td>10110974</td>
    <td>2026-05-01</td>
    <td>-43.00</td>
    <td>restaurants</td>
    </tr>
    </tbody>
    </table>
    <p><strong>Notes:</strong></p>
    <ul>
    <li>All amounts are negative, suggesting these represent expenses or money leaving the account</li>
    <li>Categories include: shopping, restaurants, transfer, and transport</li>
    <li>Data spans from October 2025 to May 2026</li>
    </ul>
    <p>The query I ran:</p>
    <div class="highlight"><pre><span></span><span class="k">SELECT</span><span class="w"> </span><span class="o">*</span><span class="w"> </span><span class="k">FROM</span><span class="w"> </span><span class="k">default</span><span class="p">.</span><span class="n">transactions</span><span class="w"> </span><span class="k">LIMIT</span><span class="w"> </span><span class="mi">5</span>
    </pre></div>
    </div>
    </div>

Finally, we ask the agent to join the tables and compute a derived metric.

.. code:: ipython3

    # Turn 4 / 4
    prompt = """
    Which account type has the highest average monthly spending per account? Only consider active accounts and exclude salary payments.
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)

.. raw:: html

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Which account type has the highest average monthly spending per account? Only consider active accounts and exclude salary payments.
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Agent</div>
    <div class="message">
    <p>🔨 Ran <code>run_query</code></p>
    <pre class="tool-x-scroll">{'input': {'query': "WITH monthly_spending AS (\n    SELECT \n        a.account_type,\n        a.account_id,\n        toYYYYMM(t.transaction_date) AS yyyymm,\n        sum(t.amount) AS monthly_amount\n    FROM default.accounts a\n    JOIN default.transactions t ON a.account_id = t.account_id\n    WHERE a.status = 'active'\n      AND t.category != 'salary'\n    GROUP BY a.account_type, a.account_id, yyyymm\n)\nSELECT \n    account_type,\n    round(avg(monthly_amount), 2) AS avg_monthly_spending\nFROM monthly_spending\nGROUP BY account_type\nORDER BY abs(avg_monthly_spending) DESC"}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">{'output': '{"columns": ["account_type", "avg_monthly_spending"], "rows": [["savings", -162.66], ["current", -162.0], ["credit_card", -155.34]]}'}
    </pre>
    <h2>📊 Results: Highest Average Monthly Spending by Account Type</h2>
    <p>Based on <strong>active accounts only</strong> and <strong>excluding salary payments</strong>:</p>
    <table>
    <thead>
    <tr>
    <th>Account Type</th>
    <th>Avg Monthly Spending (per account)</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td><strong>savings</strong></td>
    <td><strong>$162.66</strong></td>
    </tr>
    <tr>
    <td>current</td>
    <td>$162.00</td>
    </tr>
    <tr>
    <td>credit_card</td>
    <td>$155.34</td>
    </tr>
    </tbody>
    </table>
    <p>🏆 <strong>Winner: <code>savings</code> accounts</strong> have the highest average monthly spending per account at <strong>$162.66</strong>.</p>
    <hr/>
    <p><strong>Key Observations:</strong></p>
    <ul>
    <li>All amounts are negative, indicating money leaving the account (expenses)</li>
    <li>The difference between the highest (savings) and lowest (credit_card) is relatively small (~$7.32/month)</li>
    <li>Only active accounts were included in the analysis</li>
    <li>Salary payments were excluded as requested</li>
    </ul>
    <p><strong>Query used:</strong></p>
    <div class="highlight"><pre><span></span><span class="k">WITH</span><span class="w"> </span><span class="n">monthly_spending</span><span class="w"> </span><span class="k">AS</span><span class="w"> </span><span class="p">(</span>
    <span class="w">    </span><span class="k">SELECT</span><span class="w"> </span>
    <span class="w">        </span><span class="n">a</span><span class="p">.</span><span class="n">account_type</span><span class="p">,</span>
    <span class="w">        </span><span class="n">a</span><span class="p">.</span><span class="n">account_id</span><span class="p">,</span>
    <span class="w">        </span><span class="n">toYYYYMM</span><span class="p">(</span><span class="n">t</span><span class="p">.</span><span class="n">transaction_date</span><span class="p">)</span><span class="w"> </span><span class="k">AS</span><span class="w"> </span><span class="n">yyyymm</span><span class="p">,</span>
    <span class="w">        </span><span class="k">sum</span><span class="p">(</span><span class="n">t</span><span class="p">.</span><span class="n">amount</span><span class="p">)</span><span class="w"> </span><span class="k">AS</span><span class="w"> </span><span class="n">monthly_amount</span>
    <span class="w">    </span><span class="k">FROM</span><span class="w"> </span><span class="k">default</span><span class="p">.</span><span class="n">accounts</span><span class="w"> </span><span class="n">a</span>
    <span class="w">    </span><span class="k">JOIN</span><span class="w"> </span><span class="k">default</span><span class="p">.</span><span class="n">transactions</span><span class="w"> </span><span class="n">t</span><span class="w"> </span><span class="k">ON</span><span class="w"> </span><span class="n">a</span><span class="p">.</span><span class="n">account_id</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">t</span><span class="p">.</span><span class="n">account_id</span>
    <span class="w">    </span><span class="k">WHERE</span><span class="w"> </span><span class="n">a</span><span class="p">.</span><span class="n">status</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="s1">'active'</span>
    <span class="w">      </span><span class="k">AND</span><span class="w"> </span><span class="n">t</span><span class="p">.</span><span class="n">category</span><span class="w"> </span><span class="o">!=</span><span class="w"> </span><span class="s1">'salary'</span>
    <span class="w">    </span><span class="k">GROUP</span><span class="w"> </span><span class="k">BY</span><span class="w"> </span><span class="n">a</span><span class="p">.</span><span class="n">account_type</span><span class="p">,</span><span class="w"> </span><span class="n">a</span><span class="p">.</span><span class="n">account_id</span><span class="p">,</span><span class="w"> </span><span class="n">yyyymm</span>
    <span class="p">)</span>
    <span class="k">SELECT</span><span class="w"> </span>
    <span class="w">    </span><span class="n">account_type</span><span class="p">,</span>
    <span class="w">    </span><span class="n">round</span><span class="p">(</span><span class="k">avg</span><span class="p">(</span><span class="n">monthly_amount</span><span class="p">),</span><span class="w"> </span><span class="mi">2</span><span class="p">)</span><span class="w"> </span><span class="k">AS</span><span class="w"> </span><span class="n">avg_monthly_spending</span>
    <span class="k">FROM</span><span class="w"> </span><span class="n">monthly_spending</span>
    <span class="k">GROUP</span><span class="w"> </span><span class="k">BY</span><span class="w"> </span><span class="n">account_type</span>
    <span class="k">ORDER</span><span class="w"> </span><span class="k">BY</span><span class="w"> </span><span class="k">abs</span><span class="p">(</span><span class="n">avg_monthly_spending</span><span class="p">)</span><span class="w"> </span><span class="k">DESC</span>
    </pre></div>
    <p>The calculation first groups spending by account and month, then averages those monthly totals across all accounts of each type.</p>
    </div>
    </div>

The agent handles schema discovery and the follow-up reference correctly, and writes valid ClickHouse SQL for the final question, but its responses also contain errors. In the third turn, it infers the date range of the whole table from five rows of a single account. In the last turn, it averages spending only over the months in which each account has transactions, which overstates the monthly figures, draws a conclusion from differences between account types that are too small to be meaningful, and reports the amounts in dollars, although the dataset does not specify a currency. The agent does not flag any of these issues itself, which is why the generated queries and their results still need to be reviewed.

3. Conclusion
***************************************************************************************************************

In this post, we built a text-to-SQL agent with Strands Agents and deployed it on Amazon Bedrock AgentCore Runtime with private access to an open-weight LLM on Amazon SageMaker and to a ClickHouse Cloud database over AWS PrivateLink. The model weights are loaded from an S3 bucket, the model container runs with network isolation, and neither the model nor the database can be reached from the internet. The agent translates natural language questions into SQL, runs them against the database, and explains the results, with full visibility into the tool calls and their output.

This is a basic implementation intended as a starting point. In our setup, the agent reaches AWS services such as Secrets Manager and CloudWatch Logs through a NAT gateway. A stricter configuration would replace the NAT gateway with interface endpoints to each of these services, so that the subnet has no route to the internet at all, and would restrict the database credentials in Secrets Manager to the corresponding endpoint.

The example dataset also contains personally identifiable information (PII), such as customer names and dates of birth. Although the data is synthetic, a production deployment working on real customer data would need to handle this information in line with the applicable regulation. Further controls, such as SQL validation before execution and ongoing monitoring of quality, latency, errors and costs, would also be required.

The full code is available on `GitHub <https://github.com/flaviagiammarino/agentcore-clickhouse-agent/>`__.
