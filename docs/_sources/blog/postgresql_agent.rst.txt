:layout: post

.. post:: June 6, 2026
    :tags: Amazon Bedrock, Strands Agents
    :category: Agentic AI
    :author: Flavia
    :description: Build and deploy a PostgreSQL text-to-SQL agent with Amazon Bedrock and Strands Agents
    :keywords: Text-to-SQL, Amazon Bedrock, Strands Agents
    :image_src: https://machine-learning-blog.s3.eu-west-2.amazonaws.com/postgresql_agent/architecture_diagram.png
    :image_alt: Architecture diagram showing a user invoking a Strands agent on Amazon Bedrock AgentCore Runtime in a private VPC subnet where the agent queries an Amazon RDS PostgreSQL database.
    :excerpt: 1

###############################################################################################################
Deploying a text-to-SQL agent with private VPC access to RDS PostgreSQL on Amazon Bedrock AgentCore
###############################################################################################################

.. raw:: html

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/postgresql_agent/architecture_diagram.png"
        style="width:100%"
        alt="Architecture diagram showing a user invoking a Strands agent on Amazon Bedrock AgentCore Runtime in a private VPC subnet where the agent queries an Amazon RDS PostgreSQL database."
    >

1. Overview
***************************************************************************************************************

Text-to-SQL translates natural language queries into structured SQL statements, allowing users without specialized database knowledge to interactively explore and analyze data - beyond the pre-configured queries of static dashboards and reports, and without depending on technical teams to write and run custom queries.

In this post, we build a text-to-SQL agent using Strands Agents and deploy it to Amazon Bedrock AgentCore Runtime in a private VPC.
The agent accesses an Amazon RDS PostgreSQL instance in the same VPC via an MCP server and communicates with other AWS services through a NAT Gateway.

The agent translates user questions into SQL queries, executes them against the database, and explains the results in natural language.
Unlike RAG-based approaches that require re-syncing the knowledge base whenever the schema or data changes, the agent queries directly the live database, automatically incorporating any changes without manual intervention.

2. Solution
***************************************************************************************************************

The solution involves creating an RDS PostgreSQL database, updating the default VPC configuration, building the agent in Python with Strands Agents, containerizing it with Docker, pushing the image to Amazon ECR, and deploying it to Amazon Bedrock AgentCore Runtime. The implementation consists of three files:

.. code-block:: text

    agent/
    ├── agent.py              # Strands agent with database tools
    ├── Dockerfile            # Container image definition
    └── build_and_deploy.sh   # ECR image creation and AgentCore Runtime deployment

To follow along, you will need an AWS account with sufficient IAM permissions to create and manage the resources described in this post, the AWS CLI installed and configured with appropriate credentials, Docker for building and pushing the container image, and Python 3.13 with `uv <https://docs.astral.sh/uv/>`__ for project setup and dependency management.

2.1 Create the PostgreSQL database in RDS
===============================================================================================================

For this demonstration, we use two small tables: ``products`` and ``sales``,
containing product names and unit sales for three items.
We create the two tables in a PostgreSQL RDS instance with public access disabled and store the database credentials in AWS Secrets Manager.
The instance runs within the default VPC, which has three public subnets, an internet gateway, and DNS resolution and DNS hostnames enabled.

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/postgresql_agent/products_table.png"
        style="width:70%"
        alt="products Postgres table"
    >

    <p>
    <span style="font-weight:600">Figure 1:</span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">products</span>
    </code>
    <span>Postgres table.</span>
    </p>

    </div>

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/postgresql_agent/sales_table.png"
        style="width:70%"
        alt="sales Postgres table"
    >

    <p>
    <span style="font-weight:600">Figure 2:</span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">sales</span>
    </code>
    <span>Postgres table.</span>
    </p>

    </div>

2.2 Update the default VPC configuration
===============================================================================================================

To allow the agent to reach other AWS services such as Amazon Bedrock, AWS Secrets Manager and Amazon CloudWatch, we create a private subnet in the default VPC where the RDS instance is running. Since resources in a private subnet have no public IP and cannot reach the internet directly, we place a NAT Gateway in one of the existing public subnets with an Elastic IP. We then create a route table that sends all outbound traffic to the NAT Gateway and associate it with the private subnet. This gives the agent outbound internet access through the NAT Gateway while keeping it on the same private network as the RDS database. Multiple private subnets across different AZs could additionally be set up to improve agent availability. For more details on the VPC configuration, see `this article in the AWS Builders Center <https://builder.aws.com/content/2xQRB09BKuwZ7aMcLZBh4ycoTvV/deploying-amazon-bedrock-agentcore-runtime-in-a-vpc-a-step-by-step-guide>`__.

2.3 Build the agent with Strands Agents
===============================================================================================================

The agent connects to the database with the `postgres-mcp <https://github.com/crystaldba/postgres-mcp>`__ server, which includes tools for listing schemas, describing tables and running SQL queries. The agent retrieves the database user and password from AWS Secrets Manager. The secret ID, the database host, and the database name are passed as environment variables to the AgentCore Runtime by the deployment script in the next section.

We configure the ``postgres-mcp`` server to run as a subprocess via ``uv``. The database URI, including the SSL certificate path for secure connections to RDS, is passed to the subprocess as an environment variable. The MCP client communicates with the server over standard input/output.
We start the client at module level and keep it alive for the lifetime of the session.

.. warning::

    By default, the ``postgres-mcp`` server runs in ``unrestricted`` access mode, allowing both read and write access to the database. For production use, set the access mode to ``restricted`` to limit the agent to read-only queries.

The agent is initialized with Claude Sonnet 4.6 as the underlying model, the ``postgres-mcp`` tools, and a system prompt.
The agent is wrapped in a ``BedrockAgentCoreApp`` with an async streaming entrypoint that yields agent events — including not only text responses, but also tool call inputs and results — back to the caller.

.. code:: python

    import os
    import json
    import boto3
    from strands import Agent
    from strands.tools.mcp import MCPClient
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    # Create a Secrets Manager client and retrieve the database credentials
    client = boto3.client("secretsmanager")
    secret = client.get_secret_value(SecretId=os.environ["SECRET_ID"])

    # Parse the secret string and extract the connection details
    secret_string = json.loads(secret["SecretString"])
    db_config = {
        "user": secret_string["username"],
        "password": secret_string["password"],
        "host": os.environ["DB_HOST"],
        "name": os.environ["DB_NAME"],
    }

    def create_postgres_client():
        # Configure the postgres-mcp server to run as a subprocess via uv
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "postgres-mcp"],
            env={"DATABASE_URI": f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['name']}?sslmode=verify-full&sslrootcert=/app/global-bundle.pem"}
        )
        # Return an MCP client that communicates with the server over stdio
        return MCPClient(lambda: stdio_client(server_params))

    # Start the MCP client and keep it alive for the lifetime of the session
    postgres_client = create_postgres_client()
    postgres_client.__enter__()

    # Retrieve the list of tools exposed by the postgres-mcp server
    tools = postgres_client.list_tools_sync()

    # Initialize the Strands agent with the postgres tools
    agent = Agent(
        model="eu.anthropic.claude-sonnet-4-6",
        tools=tools,
        system_prompt=(
            "You are a text-to-SQL assistant with access to a PostgreSQL database. "
            "When answering questions, always inspect the database schema first to understand the available tables and columns. "
            "Generate accurate SQL queries based on the user's question and the actual schema. "
            "Return the query results in a clear, readable format. "
            "If a question cannot be answered from the available data, say so explicitly."
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

We deploy the agent using Docker. The Dockerfile uses ``uv``'s ARM64 Python 3.13 base image. We copy the ``pyproject.toml`` and ``uv.lock`` files and install the dependencies, then download the RDS CA certificate bundle required for SSL connections to PostgreSQL. Finally, we copy the agent code, expose port 8080 and run the application. The ``pyproject.toml`` and ``uv.lock`` files referenced in the Dockerfile are generated by the deployment script in the next section.

.. code:: docker

    # Use uv's ARM64 Python 3.13 base image
    FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.13-bookworm-slim

    # Set the working directory inside the container
    WORKDIR /app

    # Copy uv files
    COPY pyproject.toml uv.lock ./

    # Install dependencies
    RUN uv sync --frozen --no-cache

    # Download the RDS CA certificate bundle for SSL connections to PostgreSQL
    RUN apt-get update && apt-get install -y wget && \
        wget -O /app/global-bundle.pem https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem && \
        apt-get remove -y wget && rm -rf /var/lib/apt/lists/*

    # Copy Python modules
    COPY agent.py ./

    # Expose port
    EXPOSE 8080

    # Run application
    CMD ["uv", "run", "python", "agent.py"]

2.4 Deploy the agent to Amazon Bedrock AgentCore
===============================================================================================================

The deployment script initializes a ``uv`` project and adds the required dependencies.
This generates the ``pyproject.toml`` and ``uv.lock`` files used by the Dockerfile.
It then authenticates with Amazon ECR, creates a repository and builds and pushes the container image.
Finally, it deploys the agent to AgentCore Runtime in VPC mode, specifying the private subnet and security group for network configuration.
The ID of the secret with the database user and password, the database host, and the database name are passed as environment variables and retrieved at runtime.

.. code:: bash

    # AWS account ID
    aws_account_id="<aws-account-id>"

    # AWS region for all resources
    region="<aws-region>"

    # AgentCore Runtime name
    agent_name="<agentcore-runtime-name>"

    # ECR repository name for the container image
    repository_name="<ecr-repository-name>"

    # Secrets Manager secret ID for RDS credentials
    secret_id="<secret-name>"

    # RDS instance endpoint
    db_host="<rds-db-host>"

    # RDS database name
    db_name="<rds-db-name>"

    # IAM execution role for AgentCore Runtime
    role_name="<agentcore-runtime-role>"

    # Private subnet (routes through NAT Gateway for outbound internet access)
    subnet_id="<private-subnet-id>"

    # Security group (allows HTTPS outbound for AWS API calls and inbound from RDS)
    security_group_id="<vpc-security-group>"

    # Project setup
    uv init --name postgres-agent --description "Strands Agent for Postgres Text-to-SQL" --python 3.13 --bare
    uv lock
    uv add strands-agents fastmcp postgres-mcp boto3 bedrock-agentcore

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
      --network-configuration "{
        \"networkMode\": \"VPC\",
        \"networkModeConfig\": {
          \"subnets\": [\"${subnet_id}\"],
          \"securityGroups\": [\"${security_group_id}\"]
        }
      }" \
      --environment-variables "SECRET_ID=${secret_id},DB_HOST=${db_host},DB_NAME=${db_name}" \
      --role-arn "arn:aws:iam::${aws_account_id}:role/${role_name}" \
      --region "$region"

2.5 Test the agent in a Jupyter Notebook
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
Runtime ARN and region, create a Boto3 client for invoking the
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
                    if "content" in content["toolResult"]:
                        tool_output = content["toolResult"]["content"][0]["text"]
                        display(Markdown("🔨 Output:"))
                        print({"output": tool_output})

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
    <p>
    <p>I'm your PostgreSQL database assistant! Here's what I can help you with:</p>
    <hr/>
    <h2>🔍 Schema Exploration</h2>
    <ul>
    <li><strong>Browse your database</strong>: List schemas, tables, views, sequences, and extensions</li>
    <li><strong>Inspect objects</strong>: Get detailed info on table columns, data types, constraints, indexes, and more</li>
    </ul>
    <hr/>
    <h2>📝 SQL Query Generation &amp; Execution</h2>
    <ul>
    <li><strong>Answer questions</strong> about your data by writing and running SQL queries</li>
    <li><strong>Translate plain English</strong> into accurate SQL — just ask something like <em>"How many orders were placed last month?"</em></li>
    <li><strong>Run custom SQL</strong> queries you provide directly</li>
    </ul>
    <hr/>
    <h2>⚡ Performance &amp; Optimization</h2><ul>
    <li><strong>Explain query plans</strong>: Understand how PostgreSQL executes a query and identify bottlenecks</li>
    <li><strong>Index recommendations</strong>: Analyze specific queries or your overall workload and suggest optimal indexes</li>
    <li><strong>Top queries</strong>: Find the slowest or most resource-intensive queries in your system</li>
    </ul>
    <hr/>
    <h2>🏥 Database Health</h2><ul>
    <li><strong>Full health checks</strong> covering:<ul>
    <li>🔗 Connections &amp; utilization</li>
    <li>🗂️ Index health (invalid, duplicate, bloated indexes)</li>
    <li>🧹 Vacuum &amp; transaction ID wraparound</li>
    <li>🔄 Replication lag &amp; slots</li>
    <li>💾 Buffer cache hit rates</li>
    <li>🔢 Sequence overflow risks</li>
    <li>✅ Constraint validity</li>
    </ul>
    </li>
    </ul>
    <hr/>
    <h2>💡 Example Questions You Can Ask</h2><ul>
    <li><em>"Show me the tables in the public schema"</em></li>
    <li><em>"How many users signed up this week?"</em></li>
    <li><em>"What are the slowest queries in my database?"</em></li>
    <li><em>"Is my database healthy?"</em></li>
    <li><em>"Recommend indexes for this query: SELECT ..."</em></li>
    </ul>
    <hr/>
    <p><strong>Just ask your question in plain English, and I'll take it from there!</strong> 🚀</p>
    </div>
    </div>

We then ask the agent to list the tables in the database.
The initial connection to PostgreSQL fails due to an SSL handshake timing issue, but the agent identifies the error and retries successfully.

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
    <p>Let me look up all the schemas and their tables for you!</p>
    <p>🔨 Ran <code>list_schemas</code></p>
    <pre class="tool-x-scroll">
    {'input': {}}
    </pre>
    <p>🔨 Output:</p>
    <pre  class="tool-x-scroll">
    {'output': 'Error: consuming input failed: SSL error: unexpected eof while reading'}
    </pre>
    <p>It seems there was a connection issue. Let me try again!</p>
    <p>🔨 Ran <code>list_schemas</code></p>
    <pre class="tool-x-scroll">
    {'input': {}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">
    {'output': "[
        {
            'schema_name': 'information_schema',
            'schema_owner': 'rdsadmin',
            'schema_type': 'System Information Schema'
        },
        {
            'schema_name': 'pg_catalog',
            'schema_owner': 'rdsadmin',
            'schema_type': 'System Schema'
        },
        {
            'schema_name': 'public',
            'schema_owner': 'pg_database_owner',
            'schema_type': 'User Schema'
        }
    ]"}
    </pre>
    <p>There's one user schema — <strong>public</strong>. Let me fetch the tables in it!</p>
    <p>🔨 Ran <code>list_objects</code></p>
    <pre class="tool-x-scroll">
    {'input': {'schema_name': 'public', 'object_type': 'table'}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">
    {'output': "[
        {
            'schema': 'public',
            'name': 'products',
            'type': 'BASE TABLE'
        },
        {
            'schema': 'public',
            'name': 'sales',
            'type': 'BASE TABLE'
         }
    ]"}
    </pre>
    <p>Your database has <strong>1 user schema (<code>public</code>)</strong> with <strong>2 tables</strong>:</p>
    <table>
    <thead>
    <tr>
    <th>#</th>
    <th>Table Name</th>
    <th>Type</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>1</td>
    <td><strong>products</strong></td>
    <td>BASE TABLE</td>
    </tr>
    <tr>
    <td>2</td>
    <td><strong>sales</strong></td>
    <td>BASE TABLE</td>
    </tr>
    </tbody>
    </table>
    <p>Would you like me to:</p>
    <ul>
    <li>🔎 <strong>Inspect the details</strong> of either table (columns, data types, indexes, constraints)?</li>
    <li>📊 <strong>Query the data</strong> — e.g., row counts, sample rows, or something more specific?</li>
    </ul>
    </div>
    </div>

Next, we ask the agent a question that relies on context from the previous turn to demonstrate short-term memory capabilities.

.. code:: ipython3

    # Turn 3 / 4
    prompt = """
    Show me the data in the second table.
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)

.. raw:: html

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Show me the data in the second table.
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Agent</div>
    <div class="message">
    <p>Let me first check the structure of the <strong>sales</strong> table, then query the data — all at once!</p>
    <p>🔨 Ran <code>get_object_details</code></p>
    <pre class="tool-x-scroll">
    {'input': {'schema_name': 'public', 'object_name': 'sales'}}
    </pre>
    <p>🔨 Ran <code>execute_sql</code></p>
    <pre class="tool-x-scroll">
    {'input': {'sql': 'SELECT * FROM public.sales LIMIT 100;'}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">
    {'output': "{
        'basic': {
            'schema': 'public',
            'name': 'sales',
            'type': 'table'
        },
        'columns': [
            {
                'column': 'product_id',
                'data_type': 'character varying',
                'is_nullable': 'YES',
                'default': None
            },
            {
                'column': 'units',
                'data_type': 'integer',
                'is_nullable': 'YES',
                'default': None
            }
        ],
        'constraints': [
            {
                'name': 'sales_product_id_fkey',
                'type': 'FOREIGN KEY',
                'columns': ['product_id']
            }
        ],
        'indexes': []
    }"}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">
    {'output': "[
        {'product_id': '1', 'units': 10},
        {'product_id': '2', 'units': 20},
        {'product_id': '3', 'units': 30}
    ]"}
    </pre>
    <p>Here's the data in the <strong>sales</strong> table:</p>
    <table>
    <thead>
    <tr>
    <th>product_id</th>
    <th>units</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>1</td>
    <td>10</td>
    </tr>
    <tr>
    <td>2</td>
    <td>20</td>
    </tr>
    <tr>
    <td>3</td>
    <td>30</td>
    </tr>
    </tbody>
    </table>
    <p><strong>Table Details:</strong></p>
    <ul>
    <li><strong>Columns:</strong> <code>product_id</code> (varchar), <code>units</code> (integer)</li>
    <li><strong>Constraints:</strong> <code>product_id</code> is a <strong>foreign key</strong> referencing the <code>products</code> table</li>
    <li><strong>Rows returned:</strong> 3</li>
    </ul>
    <p>Would you like me to <strong>join this with the products table</strong> to see more details, or do any further analysis on this data?</p>
    </div>
    </div>

Finally, we ask the agent to join the two tables and compute a simple derived metric.

.. code:: ipython3

    # Turn 4 / 4
    prompt = """
    Combine the two tables and add the percentage of total sales for each product.
    """
    messages = get_streaming_response(prompt, session_id)
    print_messages(prompt, messages)

.. raw:: html

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Combine the two tables and add the percentage of total sales for each product.
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Agent</div>
    <div class="message">
    <p>🔨 Ran <code>get_object_details</code></p>
    <pre class="tool-x-scroll">
    {'input': {'schema_name': 'public', 'object_name': 'products'}}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">
    {'output': "{
        'basic': {
            'schema': 'public',
            'name': 'products',
            'type': 'table'
        },
        'columns': [
            {
                'column': 'product_id',
                'data_type': 'character varying',
                'is_nullable': 'NO',
                'default': None
            },
            {
                'column': 'product_name',
                'data_type': 'character varying',
                'is_nullable': 'YES',
                'default': None
            }
        ],
        'constraints': [
            {
                'name': 'products_product_id_not_null',
                'type': 'CHECK',
                'columns': []
            },
            {
                'name': 'products_pkey',
                'type': 'PRIMARY KEY',
                'columns': ['product_id']
            }
        ],
        'indexes': [
            {
                'name': 'products_pkey',
                'definition': 'CREATE UNIQUE INDEX products_pkey ON public.products USING btree (product_id)'
             }
        ]
    }"}
    </pre>
    <p>Got the schema! Now let me write and run the combined query.</p>
    <p>🔨 Ran <code>execute_sql</code></p>
    <pre class="tool-x-scroll">
    {'input': {
        'sql':
            'SELECT
                p.product_id,
                p.product_name,
                s.units,
                ROUND(s.units * 100.0 / SUM(s.units) OVER (), 2) AS percentage_of_total_sales
            FROM
                public.products p
            JOIN
                public.sales s
            ON
                p.product_id = s.product_id
            ORDER BY
                p.product_id;'
    }}
    </pre>
    <p>🔨 Output:</p>
    <pre class="tool-x-scroll">
    {'output': "[
        {
            'product_id': '1',
            'product_name': 'Product A',
            'units': 10,
            'percentage_of_total_sales': Decimal('16.67')
        },
        {
            'product_id': '2',
            'product_name': 'Product B',
            'units': 20,
            'percentage_of_total_sales': Decimal('33.33')
        },
        {
            'product_id': '3',
            'product_name': 'Product C',
            'units': 30,
            'percentage_of_total_sales': Decimal('50.00')
        }
    ]"}
    </pre>
    <p>Here's the combined table with the percentage of total sales for each product:</p>
    <table>
    <thead>
    <tr>
    <th>Product ID</th>
    <th>Product Name</th>
    <th>Units</th>
    <th>% of Total Sales</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>1</td>
    <td>Product A</td>
    <td>10</td>
    <td>16.67%</td>
    </tr>
    <tr>
    <td>2</td>
    <td>Product B</td>
    <td>20</td>
    <td>33.33%</td>
    </tr>
    <tr>
    <td>3</td>
    <td>Product C</td>
    <td>30</td>
    <td>50.00%</td>
    </tr>
    <tr>
    <td></td>
    <td><strong>Total</strong></td>
    <td><strong>60</strong></td>
    <td><strong>100%</strong></td>
    </tr>
    </tbody>
    </table>
    <p><strong>Key Insights:</strong></p>
    <ul>
    <li>🥇 <strong>Product C</strong> is the top seller, accounting for <strong>half of all sales (50%)</strong></li>
    <li>🥈 <strong>Product B</strong> comes in second at <strong>33.33%</strong></li>
    <li>🥉 <strong>Product A</strong> has the smallest share at <strong>16.67%</strong></li>
    </ul>
    <p>Would you like any further analysis, such as visualizations, filtering, or additional metrics?</p>
    </div>
    </div>

To avoid ongoing charges, delete the AgentCore runtime, ECR repository, RDS instance, NAT Gateway, Elastic IP, private subnet and route table after testing.

3. Conclusion
***************************************************************************************************************

In this post, we deployed a text-to-SQL agent using Strands Agents on Amazon Bedrock AgentCore Runtime with private VPC access to an RDS PostgreSQL database. The agent translates natural language questions into SQL queries, executes them against the database, and explains the results - with full visibility into the reasoning steps and tool calls.

This is a basic implementation intended as a starting point. A production deployment would require additional controls, such as SQL validation to verify generated queries for correctness before execution, access controls to ensure that users can only query data they are authorized to access, caching to reduce latency and model invocation costs for frequent queries, and ongoing monitoring of quality, latency, throughput, errors and costs. For further discussion of implementing production text-to-SQL solutions in Amazon Bedrock, see `this article in the AWS Blog <https://aws.amazon.com/blogs/machine-learning/text-to-sql-solution-powered-by-amazon-bedrock/>`__.

You can download the full code from our `GitHub repository <https://github.com/flaviagiammarino/machine-learning-blog/tree/main/postgresql_agent/>`__.
