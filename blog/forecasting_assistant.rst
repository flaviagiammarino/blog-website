:layout: post

.. post:: March 15, 2026
    :tags: Time Series Forecasting, Large Language Models, Amazon Chronos, Amazon Bedrock
    :category: Time Series Foundation Models
    :author: Flavia
    :description: Build a time series forecasting assistant with ClickHouse, Amazon Bedrock, and LibreChat
    :keywords: Time Series Forecasting, Large Language Models, Amazon Chronos, Amazon Bedrock, LibreChat, ClickHouse
    :image_src: https://machine-learning-blog.s3.eu-west-2.amazonaws.com/forecasting-assistant/architecture_diagram.png
    :image_alt: Architecture diagram showing a user interacting with LibreChat, which routes requests to Claude Sonnet 4.6 on Amazon Bedrock to query a ClickHouse database and generate forecasts using Chronos on Amazon Bedrock
    :excerpt: 1

###############################################################################################################
Building a time series forecasting assistant with Amazon Bedrock and LibreChat
###############################################################################################################

.. raw:: html

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/forecasting-assistant/architecture_diagram.png"
        style="width:100%"
        alt="Architecture diagram showing a user interacting with LibreChat, which routes requests to Claude Sonnet 4.6 on Amazon Bedrock to query a ClickHouse database and generate forecasts using Chronos on Amazon Bedrock"
    >

1. Overview
***************************************************************************************************************

Most forecasting systems expose their outputs through dashboards and reports, limiting the user's interaction and visibility
to predefined queries anticipated at development time. In this post, we address this limitation by building
a conversational time series forecasting assistant using ClickHouse, Amazon Bedrock, and `LibreChat <https://www.librechat.ai/>`__.
The assistant allows the user to explore the underlying data, select specific time series, and adjust forecast parameters - such as
quantile levels and prediction horizons - through natural language.

The solution is built around the `Model Context Protocol (MCP) <https://modelcontextprotocol.io>`__,
which defines a common interface for connecting language models to external tools and services, and relies on three servers:
the `ClickHouse MCP server <https://github.com/ClickHouse/mcp-clickhouse>`__ for retrieving the time series,
a custom MCP server that generates probabilistic time series forecasts using Amazon Chronos `[1, 2] <#references>`_
deployed on Amazon Bedrock, and an additional custom MCP server that creates interactive Plotly charts of the time series and their forecasts.
We use LibreChat as the chat UI, with Claude Sonnet 4.6 on Amazon Bedrock as the orchestrating model.

For simplicity, this demonstration uses artificially generated sales data for three products.
However, as Amazon Chronos performs zero-shot forecasting without fine-tuning, this solution can be applied
to any time series dataset. In the rest of the post, we walk through the implementation of the three MCP servers and
the assistant configuration. We also illustrate a typical conversation with the assistant, which is shown in the video below;
the transcript is reported at the end of this post.

.. raw:: html

    <video width="100%" controls style="border-radius: 0.5rem; mask-image: radial-gradient(white, white); -webkit-mask-image: -webkit-radial-gradient(white, white);">
        <source src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/forecasting-assistant/video.mp4" type="video/mp4">
        A conversation with a time series forecasting assistant built using Amazon Bedrock and LibreChat.
        The assistant retrieves the data from ClickHouse and uses Amazon Chronos deployed on Amazon Bedrock to generate forecasts.
    </video>

2. Solution
***************************************************************************************************************

For this example, we use synthetic hourly sales data for three products: Product A, Product B and Product C.
The time series span 90 days, and are stored in two ClickHouse tables: ``products``, which maps product IDs to names,
and ``sales``, which records hourly units sold per product.

.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/forecasting-assistant/products.png"
        style="width:90%"
        alt="products ClickHouse table"
    >

    <p>
    <span style="font-weight:600">Figure 1:</span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">products</span>
    </code>
    <span>ClickHouse table.</span>
    </p>

    </div>


.. raw:: html

    <div style="margin: 2em 0em 2em 0em">

    <img
        src="https://machine-learning-blog.s3.eu-west-2.amazonaws.com/forecasting-assistant/sales.png"
        style="width:90%"
        alt="Last 9 rows of sales ClickHouse table"
    >

    <p>
    <span style="font-weight:600">Figure 2:</span>
    <span>Last 9 rows of </span>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">sales</span>
    </code>
    <span>ClickHouse table.</span>
    </p>

    </div>

2.1 Set up the MCP servers
===============================================================================================================

We start by cloning the `LibreChat GitHub repository <https://github.com/danny-avila/LibreChat>`__ and configuring
Claude Sonnet 4.6 on Amazon Bedrock as the underlying model. We use the ``eu.anthropic.claude-sonnet-4-6``
cross-region inference profile in ``eu-west-1``. As outlined in the `LibreChat documentation <https://www.librechat.ai/docs/configuration/pre_configured_ai/bedrock>`__,
we update the following environment variables in the ``.env`` file:

.. code-block:: text

    BEDROCK_AWS_DEFAULT_REGION=eu-west-1
    BEDROCK_AWS_MODELS=eu.anthropic.claude-sonnet-4-6
    BEDROCK_AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
    BEDROCK_AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

We also include the model's inference profile ARN in ``librechat.yaml``:

.. code-block:: yaml

    endpoints:
      bedrock:
        inferenceProfiles:
          "eu.anthropic.claude-sonnet-4-6": "arn:aws:bedrock:eu-west-1:<AWS_ACCOUNT_ID>:inference-profile/eu.anthropic.claude-sonnet-4-6"

2.1.1 Configure the ClickHouse MCP server
---------------------------------------------------------------------------------------------------------------

Next, we configure the ClickHouse MCP server. Following `ClickHouse documentation <https://clickhouse.com/docs/use-cases/AI/MCP/librechat>`__,
we create a ``docker-compose.override.yml`` file and add the following configuration to it:

.. code-block:: yaml

    services:
      api:
        volumes:
          - ./librechat.yaml:/app/librechat.yaml
      mcp-clickhouse:
        image: mcp/clickhouse
        container_name: mcp-clickhouse
        ports:
          - 8001:8000
        extra_hosts:
          - "host.docker.internal:host-gateway"
        environment:
          - CLICKHOUSE_HOST=<CLICKHOUSE_HOST>
          - CLICKHOUSE_USER=<CLICKHOUSE_USER>
          - CLICKHOUSE_PASSWORD=<CLICKHOUSE_PASSWORD>
          - CLICKHOUSE_MCP_SERVER_TRANSPORT=sse
          - CLICKHOUSE_MCP_BIND_HOST=0.0.0.0

We also register the ClickHouse MCP server in ``librechat.yaml`` to run on port 8001:

.. code-block:: yaml

    mcpServers:
      clickhouse-playground:
        type: sse
        url: http://host.docker.internal:8001/sse

2.1.2 Create the forecasting MCP server
---------------------------------------------------------------------------------------------------------------

After that, we create the forecasting MCP server using the `FastMCP <https://github.com/jlowin/fastmcp>`__ library.
The server exposes a ``generate_forecasts`` tool that takes a list of historical values, a prediction length, and a
list of quantile levels, and returns probabilistic time series forecasts by invoking the Chronos Bedrock endpoint
via ``boto3``. For instructions on deploying Chronos on Bedrock, we refer to
`our previous blog post <https://flaviagiammarino.com/blog/chronos_bedrock.html#create-the-bedrock-endpoint>`__.

We configure the forecasting MCP server in the ``docker-compose.override.yml`` file
that we previously created for configuring the ClickHouse MCP server:

.. code-block:: yaml

    services:
      <clickhouse-mcp-configuration> # Added in Section 2.1.1
      chronos-forecasting:
        build:
          context: ./chronos-forecasting
        ports:
          - "8002:8002"
        environment:
          - AWS_DEFAULT_REGION=eu-west-1
          - AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
          - AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

We also register the forecasting MCP server in ``librechat.yaml`` to run on port 8002:

.. code-block:: yaml

    mcpServers:
      <clickhouse-mcp-server> # Added in Section 2.1.1
      chronos-forecasting:
        type: sse
        url: http://host.docker.internal:8002/sse

The code used for building the Docker image of the forecasting MCP server is reported below.

.. raw:: html

    <p>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">server.py</span>
    </code>
    </p>

.. code:: python

    import json
    import boto3
    from mcp.server.fastmcp import FastMCP

    # Create the FastMCP server
    mcp = FastMCP(
        name="chronos-forecasting",
        host="0.0.0.0",
        port=8002
    )

    # Register the tool with the FastMCP server
    @mcp.tool()
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
        # Create the Bedrock client
        bedrock_runtime_client = boto3.client("bedrock-runtime")

        # Invoke the Bedrock endpoint
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

        # Extract the forecasts
        forecasts = json.loads(response["body"].read()).get("predictions")[0]

        # Return the forecasts
        return forecasts


    # Run the FastMCP server with SSE transport
    if __name__ == "__main__":
        mcp.run(transport="sse")

.. raw:: html

    <p>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">requirements.txt</span>
    </code>
    </p>

.. code-block:: text

    mcp
    boto3

.. raw:: html

    <p>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">Dockerfile</span>
    </code>
    </p>

.. code-block:: docker

    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY server.py .
    CMD ["python", "server.py"]

.. _visualization-mcp:

2.1.3 Create the data visualization MCP server
---------------------------------------------------------------------------------------------------------------

The data visualization MCP server exposes a ``visualize_data`` tool that generates interactive Plotly charts
from historical data and forecast outputs. Charts are saved as self-contained HTML files and served over HTTP
on port 8004. The tool returns the URL of the interactive HTML chart, that LibreChat renders directly in the
`Artifacts <https://www.librechat.ai/docs/features/artifacts>`__ panel.

.. note::

    The data visualization MCP server is not strictly necessary, as the language model can generate the charts directly.
    However, a dedicated server ensures consistent styling and reproducible outputs, which cannot be guaranteed
    when the language model generates the charts on its own. This is particularly desirable when using the assistant
    for reporting purposes.

Like the forecasting server, the data visualization server is built using FastMCP.
As we did for the forecasting server, we configure it in ``docker-compose.override.yml``:

.. code-block:: yaml

    services:
      <clickhouse-mcp-configuration> # Added in Section 2.1.1
      <forecasting-mcp-configuration> # Added in Section 2.1.2
      data-visualization:
        build:
          context: ./data-visualization
        ports:
          - "8003:8003"
          - "8004:8004"
        volumes:
          - ./plots:/plots

We also register the data visualization MCP server in ``librechat.yaml`` to run on port 8003:

.. code-block:: yaml

    mcpServers:
      <clickhouse-mcp-server> # Added in Section 2.1.1
      <forecasting-mcp-server> # Added in Section 2.1.2
      data-visualization:
        type: sse
        url: http://host.docker.internal:8003/sse

The code used for building the Docker image of the data visualization MCP server is reported below.

.. raw:: html

    <p>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">server.py</span>
    </code>
    </p>

.. code:: python

    import os
    import uuid
    import threading
    import http.server
    import functools
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from mcp.server.fastmcp import FastMCP

    # Create the FastMCP server
    mcp = FastMCP(
        name="data-visualization",
        host="0.0.0.0",
        port=8003
    )

    # Register the tool with the FastMCP server
    @mcp.tool()
    def visualize_data(
        inputs: dict
    ) -> str:
        """
        Plot one or more time series with optional forecasts and return
        the URL of the interactive HTML chart.

        Parameters
        ===============================================================================
        inputs : dict
            A dictionary with the following keys:

            "data" (required) : dict
                A dictionary where each key is a series name and each value is the
                raw output of a ClickHouse query, with the following fields:
                    - "columns" : list of strings, must contain "timestamp" and one value column
                    - "rows"    : list of [timestamp_str, float] pairs

                Example:
                {
                    "series_1": {
                        "columns": ["timestamp", "<VALUE>"],
                        "rows": [
                            ["2026-01-01", 1.0],
                            ["2026-01-02", 2.0]
                        ]
                    },
                    "series_2": {
                        "columns": ["timestamp", "<VALUE>"],
                        "rows": [
                            ["2026-01-03", 3.0],
                            ["2026-01-04", 4.0]
                        ]
                    },
                }

            "forecasts" (optional) : dict
                Forecasts for the same time series in "data". Each key is a series
                name matching a key in "data", and each value is a dictionary with
                the following fields:
                    - "timestamp"  : list of strings representing datetimes
                    - "mean"       : list of floats (mean forecast)
                    - "<quantile>" : list of floats for each quantile level, e.g.
                                     "0.05" and "0.95" for a 90% prediction interval.

                Example:
                {
                    "series_1": {
                        "timestamp": ["2026-01-01", "2026-01-02"],
                        "mean": [1.0, 2.0],
                        "0.1": [0.5, 1.5],
                        "0.5": [1.0, 2.0],
                        "0.9": [1.5, 2.5],
                    },
                    "series_2": {
                        "timestamp": ["2026-01-03", "2026-01-04"],
                        "mean": [3.0, 4.0],
                        "0.1": [2.5, 3.5],
                        "0.5": [3.0, 4.0],
                        "0.9": [3.5, 4.5],
                    },
                }

        Returns
        ===============================================================================
        str
            The URL of the interactive HTML chart.
        """
        # Extract the data and forecasts
        data = inputs["data"]
        forecasts = inputs.get("forecasts", {})

        # Parse the data
        parsed_data = {}
        for series, query_result in data.items():
            value_col = [c for c in query_result["columns"] if c != "timestamp"][0]
            idx = query_result["columns"].index
            parsed_data[series] = {
                "timestamp": [row[idx("timestamp")] for row in query_result["rows"]],
                "values": [row[idx(value_col)] for row in query_result["rows"]]
            }

        # Create the figure
        fig = make_subplots(
            rows=len(data),
            subplot_titles=list(data.keys())
        )

        # Update the figure layout
        fig.update_layout(
            height=250 * len(data),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=50, b=50, l=50, r=50),
            hovermode="x unified",
            hoverlabel=dict(
                namelength=-1
            ),
            legend=dict(
                font=dict(
                    color="#24292f",
                    size=12
                ),
            )
        )

        # Update the subplots titles
        fig.update_annotations(
            font=dict(
                color="#24292f",
                size=14
            ),
        )

        # Generate the subplots
        for i, series in enumerate(data):
            # Plot the forecasts
            if series in forecasts:
                # Extract the predicted quantiles
                q = sorted([float(k) for k in forecasts[series] if k not in ("mean", "timestamp")])

                # Extract the lower and upper bound of the prediction interval
                q_min, q_max = q[0], q[-1]

                # Plot the upper bound of the prediction interval
                fig.add_trace(
                    go.Scatter(
                        x=forecasts[series]["timestamp"],
                        y=forecasts[series][str(q_max)],
                        name=f"Predicted Q{q_max:,.1%}",
                        hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                        showlegend=False,
                        mode="lines",
                        line=dict(
                            width=0.5,
                            color="#c2e5ff",
                        ),
                    ),
                    row=i + 1,
                    col=1
                )

                # Plot the lower bound of the prediction interval
                fig.add_trace(
                    go.Scatter(
                        x=forecasts[series]["timestamp"],
                        y=forecasts[series][str(q_min)],
                        name=f"Predicted Q{q_min:,.1%}",
                        hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                        showlegend=False,
                        mode="lines",
                        line=dict(
                            width=0.5,
                            color="#c2e5ff",
                        ),
                        fillcolor="#c2e5ff",
                        fill="tonexty",
                    ),
                    row=i + 1,
                    col=1
                )

                # Plot the predicted median if available, otherwise fall back to the predicted mean
                fig.add_trace(
                    go.Scatter(
                        x=forecasts[series]["timestamp"],
                        y=forecasts[series]["0.5" if 0.5 in q else "mean"],
                        name=f"Predicted {'Median' if 0.5 in q else 'Mean'}",
                        hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                        showlegend=i == 0,
                        mode="lines",
                        line=dict(
                            color="#0588f0",
                            width=1,
                            dash="dot"
                        )
                    ),
                    row=i + 1,
                    col=1
                )

            # Plot the data
            fig.add_trace(
                go.Scatter(
                    x=parsed_data[series]["timestamp"],
                    y=parsed_data[series]["values"],
                    name="Historical Data",
                    hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                    mode="lines",
                    showlegend=i == 0,
                    line=dict(
                        color="#838383",
                        width=1
                    )
                ),
                row=i + 1,
                col=1
            )

            # Update the subplot's x-axis
            fig.update_xaxes(
                type="date",
                tickformat="%b %d %Y<br>(%a) %H:%M",
                tickangle=0,
                mirror=True,
                linecolor="#cecece",
                gridcolor="#e8e8e8",
                gridwidth=0.5,
                tickfont=dict(
                    color="#24292f",
                    size=10
                ),
                row=i + 1,
                col=1
            )

            # Update the subplot's y-axis
            fig.update_yaxes(
                tickformat=",.0f",
                mirror=True,
                linecolor="#cecece",
                gridcolor="#e8e8e8",
                gridwidth=0.5,
                tickfont=dict(
                    color="#24292f",
                    size=10
                ),
                row=i + 1,
                col=1
            )

        # Save the figure to an HTML file
        os.makedirs("/plots", exist_ok=True)
        filename = f"plot_{uuid.uuid4().hex}.html"
        fig.write_html(f"/plots/{filename}", full_html=True, include_plotlyjs="cdn")

        # Return the URL of the HTML file
        return f"http://localhost:8004/{filename}"


    if __name__ == "__main__":
        # Serve the /plots directory over HTTP on port 8004
        def _serve_plots():
            handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="/plots")
            with http.server.HTTPServer(("0.0.0.0", 8004), handler) as httpd:
                httpd.serve_forever()


        # Start the HTTP server
        threading.Thread(target=_serve_plots, daemon=True).start()

        # Run the FastMCP server with SSE transport
        mcp.run(transport="sse")

.. raw:: html

    <p>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">requirements.txt</span>
    </code>
    </p>

.. code-block:: text

    mcp
    plotly

.. raw:: html

    <p>
    <code class="docutils literal notranslate">
    <span class="pre" style="font-weight:600">Dockerfile</span>
    </code>
    </p>

.. code-block:: docker

    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY server.py .
    CMD ["python", "server.py"]

2.2 Configure the assistant
===============================================================================================================

.. raw:: html

    <p>
    Finally, we configure the assistant in <code>librechat.yaml</code> using a
    <a href="https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/model_specs">LibreChat <code>modelSpecs</code> object</a>
    that combines the Bedrock model, the three MCP servers, and a system prompt.
    </p>

.. code-block:: yaml

    modelSpecs:
      list:
        - name: "forecasting-assistant"
          label: "Forecasting Assistant"
          default: true
          artifacts: true
          mcpServers:
            - "clickhouse-playground"
            - "chronos-forecasting"
            - "data-visualization"
          preset:
            endpoint: "bedrock"
            model: "eu.anthropic.claude-sonnet-4-6"
            region: "eu-west-1"
            promptPrefix: |
              You are a time series forecasting assistant.
              Do not alter the time frequency of the data unless the user explicitly requests it.
              To visualize data, always use the visualize_data tool. Never write code to generate plots.
              When visualizing multiple series, always pass them together to the visualize_data tool in a single call,
              as it generates subplots automatically.
              Always render the URL returned by the visualize_data tool as an artifact using this exact format:
              :::artifact{identifier="<IDENTIFIER>" type="text/html" title="<TITLE>"}
              \`\`\`
              <!DOCTYPE html>
              <html>
              <body>
                <iframe src="<URL>" width="100%" height="800px" frameborder="0"></iframe>
              </body>
              </html>
              \`\`\`
              :::
              Never return the raw URL and never ask the user to open the URL themselves.

2.3 Chat with the assistant
===============================================================================================================

To demonstrate a typical interaction, we walk through a short conversation with the assistant covering data
exploration, visualization, and forecasting. We start by asking which time series are available in the database,
then request a plot of Product A's hourly sales over the most recent week, use that data as context to forecast
the next 48 hours, and finally repeat the analysis for Products B and C, visualizing all three series and their
forecasts in a single chart.

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
    <div class="ai-name">Assistant</div>
    <div class="message">
    <details>
    <summary>Thoughts</summary>
    <pre class="thought">
    The user is asking what I can help them with. I should explain my capabilities as a time series forecasting assistant.
    </pre>
    </details>
    <p>
    I'm a <strong>time series forecasting assistant</strong> with the ability to help you with the following:
    </p>
    <h3>📊 Data Exploration</h3>
    <ul>
    <li><strong>Browse databases and tables</strong> in ClickHouse to help you find relevant data</li>
    <li><strong>Query and summarize</strong> time series data (e.g., sales, metrics, sensor readings, etc.)</li>
    </ul>
    <h3>🔮 Forecasting</h3>
    <ul>
    <li><strong>Generate probabilistic forecasts</strong> using Chronos, a state-of-the-art AI forecasting model</li>
    <li>Predict <strong>future values</strong> for any time series you have stored or provide</li>
    <li>Produce <strong>quantile forecasts</strong> (e.g., 10th–90th percentile) to understand uncertainty and risk</li>
    </ul>
    <h3>📈 Visualization</h3>
    <ul>
    <li><strong>Plot historical time series</strong> data interactively</li>
    <li><strong>Overlay forecasts</strong> with confidence intervals on top of historical data</li>
    <li>Visualize <strong>multiple series</strong> side by side for comparison</li>
    </ul>
    <h3>💡 Example Use Cases</h3>
    <ul>
    <li><em>"Show me the available tables in my database"</em></li>
    <li><em>"Plot the daily sales for the last 6 months"</em></li>
    <li><em>"Forecast the next 30 days of website traffic with a 90% confidence interval"</em></li>
    <li><em>"Compare actual vs. forecasted revenue for multiple products"</em></li>
    </ul>

    <p>
    To get started, you can ask me to <strong>explore your data</strong>, <strong>run a query</strong>,
    or <strong>generate a forecast</strong>. What would you like to do?
    </p>
    </div>
    </div>

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Which time series do we have in the database?
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Assistant</div>
    <div class="message">
    <details>
    <summary>Thoughts</summary>
    <pre class="thought">
    The user wants to know what time series data is available in the database. Let me start by listing the databases, then explore the tables within them.
    </pre>
    </details>
    <p>Let me start by exploring the available databases!</p>

    <details>
    <summary>Ran list_databases</summary>
    <p><strong>Assistant sent this info to clickhouse-playground</strong></p>
    <code>
    <pre class="tool">
    {}
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    [
      "INFORMATION_SCHEMA",
      "default",
      "information_schema",
      "system"
    ]
    </pre>
    </code>
    </details>

    <details>
    <summary>Ran list_tables</summary>
    <p><strong>Assistant sent this info to clickhouse-playground</strong></p>
    <code>
    <pre class="tool">
    {
      "database": "default"
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    [
      {
        "database": "default",
        "name": "products",
        "engine": "SharedMergeTree",
        "create_table_query": "CREATE TABLE default.products (`product_id` Int32, `product_name` String) ENGINE = SharedMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}') ORDER BY product_id SETTINGS index_granularity = 8192",
        "dependencies_database": [],
        "dependencies_table": [],
        "engine_full": "SharedMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}') ORDER BY product_id SETTINGS index_granularity = 8192",
        "sorting_key": "product_id",
        "primary_key": "product_id",
        "total_rows": 3,
        "total_bytes": 373,
        "total_bytes_uncompressed": 130,
        "parts": 1,
        "active_parts": 1,
        "total_marks": 2,
        "comment": "",
        "columns": [
          {
            "database": "default",
            "table": "products",
            "name": "product_id",
            "column_type": "Int32",
            "default_kind": "",
            "default_expression": "",
            "comment": ""
          },
          {
            "database": "default",
            "table": "products",
            "name": "product_name",
            "column_type": "String",
            "default_kind": "",
            "default_expression": "",
            "comment": ""
          }
        ]
      },
      {
        "database": "default",
        "name": "sales",
        "engine": "SharedMergeTree",
        "create_table_query": "CREATE TABLE default.sales (`timestamp` DateTime64(3), `product_id` Int32, `units_sold` Int32) ENGINE = SharedMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}') ORDER BY timestamp SETTINGS index_granularity = 8192",
        "dependencies_database": [],
        "dependencies_table": [],
        "engine_full": "SharedMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}') ORDER BY timestamp SETTINGS index_granularity = 8192",
        "sorting_key": "timestamp",
        "primary_key": "timestamp",
        "total_rows": 6480,
        "total_bytes": 13825,
        "total_bytes_uncompressed": 103808,
        "parts": 1,
        "active_parts": 1,
        "total_marks": 2,
        "comment": "",
        "columns": [
          {
            "database": "default",
            "table": "sales",
            "name": "timestamp",
            "column_type": "DateTime64(3)",
            "default_kind": "",
            "default_expression": "",
            "comment": ""
          },
          {
            "database": "default",
            "table": "sales",
            "name": "product_id",
            "column_type": "Int32",
            "default_kind": "",
            "default_expression": "",
            "comment": ""
          },
          {
            "database": "default",
            "table": "sales",
            "name": "units_sold",
            "column_type": "Int32",
            "default_kind": "",
            "default_expression": "",
            "comment": ""
          }
        ]
      }
    ]
    </pre>
    </code>
    </details>

    <details>
    <summary>Ran select_query</summary>
    <p><strong>Assistant sent this info to clickhouse-playground</strong></p>
    <code>
    <pre class="tool">
    {
      "query": "SELECT * FROM default.products"
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "columns": [
        "product_id",
        "product_name"
      ],
      "rows": [
        [
          1,
          "Product A"
        ],
        [
          2,
          "Product B"
        ],
        [
          3,
          "Product C"
        ]
      ]
    }
    </pre></code>
    </details>

    <details>
    <summary>Ran select_query</summary>
    <p><strong>Assistant sent this info to clickhouse-playground</strong></p>
    <code>
    <pre class="tool">
    {
      "query": "\nSELECT \n    min(timestamp) AS start_date,\n    max(timestamp) AS end_date,\n    count() AS total_rows\nFROM default.sales\n"
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "columns": [
        "start_date",
        "end_date",
        "total_rows"
      ],
      "rows": [
        [
          "2025-12-15T16:00:00",
          "2026-03-15T15:00:00",
          6480
        ]
      ]
    }
    </pre>
    </code>
    </details>

    <p>
    Here's what we have in the database:
    </p>
    <h3>📦 Tables in the <code>default</code> database</h3>
    <h4>1. <code>sales</code> — <strong>Main time series table</strong></h4>
    <ul>
    <li><strong>Timestamp range:</strong> Dec 15, 2025 → Mar 15, 2026 (~3 months)</li>
    <li><strong>Total rows:</strong> 6,480</li>
    <li><strong>Columns:</strong>
    <ul>
    <li><code>timestamp</code> (DateTime) — time of the observation</li>
    <li><code>product_id</code> (Int32) — links to the products table</li>
    <li><code>units_sold</code> (Int32) — the time series value</li>
    </ul>
    </li>
    </ul>
    <h4>2. <code>products</code> — <strong>Dimension/lookup table</strong></h4>
    <ul>
    <li>3 products:
    <ul>
    <li><strong>Product A</strong> (id: 1)</li>
    <li><strong>Product B</strong> (id: 2)</li>
    <li><strong>Product C</strong> (id: 3)</li>
    </ul>
    </li>
    </ul>
    <h3>🔍 Summary</h3>
    <p>You have <strong>3 time series</strong> — one per product — tracking <strong>daily (or sub-daily) units sold</strong> over roughly 3 months.</p>
    <p>Would you like me to:</p>
    <ul>
    <li>📈 <strong>Visualize</strong> the sales for one or all products?</li>
    <li>🔮 <strong>Forecast</strong> future sales for any product?</li>
    <li>🔎 <strong>Explore</strong> the data in more detail (e.g., frequency, trends)?</li>
    </ul>
    </div>
    </div>

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Can we plot the time series of Product A over the most recent week?
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Assistant</div>
    <div class="message">
    <details>
    <summary>Thoughts</summary>
    <pre class="thought">
    The user wants to plot the time series of Product A over the most recent week. Let me query the sales data for Product A over the most recent week.
    The data ends at 2026-03-15, so the most recent week would be from 2026-03-08 to 2026-03-15. Let me query accordingly.
    </pre>
    </details>
    <p>Sure! Let me first query the data for Product A over the most recent week.</p>
    <details>
    <summary>Ran select_query</summary>
    <p><strong>Assistant sent this info to clickhouse-playground</strong></p>
    <code>
    <pre class="tool">
    {
      "query": "\nSELECT timestamp, units_sold\nFROM default.sales\nWHERE product_id = 1\n  AND timestamp >= (SELECT max(timestamp) - INTERVAL 7 DAY FROM default.sales)\nORDER BY timestamp ASC\n"
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "columns": [
        "timestamp",
        "units_sold"
      ],
      "rows": [
        [
          "2026-03-08T15:00:00",
          21
        ],
        [
          "2026-03-08T16:00:00",
          20
        ],
        [
          "2026-03-08T17:00:00",
          21
        ],
        [
          "2026-03-08T18:00:00",
          25
        ],
        [
          "2026-03-08T19:00:00",
          23
        ],
        [
          "2026-03-08T20:00:00",
          24
        ],
        [
          "2026-03-08T21:00:00",
          26
        ],
        [
          "2026-03-08T22:00:00",
          25
        ],
        [
          "2026-03-08T23:00:00",
          17
        ],
        [
          "2026-03-09T00:00:00",
          13
        ],
        [
          "2026-03-09T01:00:00",
          7
        ],
        [
          "2026-03-09T02:00:00",
          4
        ],
        [
          "2026-03-09T03:00:00",
          2
        ],
        [
          "2026-03-09T04:00:00",
          7
        ],
        [
          "2026-03-09T05:00:00",
          14
        ],
        [
          "2026-03-09T06:00:00",
          22
        ],
        [
          "2026-03-09T07:00:00",
          26
        ],
        [
          "2026-03-09T08:00:00",
          35
        ],
        [
          "2026-03-09T09:00:00",
          37
        ],
        [
          "2026-03-09T10:00:00",
          39
        ],
        [
          "2026-03-09T11:00:00",
          36
        ],
        [
          "2026-03-09T12:00:00",
          32
        ],
        [
          "2026-03-09T13:00:00",
          27
        ],
        [
          "2026-03-09T14:00:00",
          23
        ],
        [
          "2026-03-09T15:00:00",
          20
        ],
        [
          "2026-03-09T16:00:00",
          19
        ],
        [
          "2026-03-09T17:00:00",
          20
        ],
        [
          "2026-03-09T18:00:00",
          22
        ],
        [
          "2026-03-09T19:00:00",
          25
        ],
        [
          "2026-03-09T20:00:00",
          26
        ],
        [
          "2026-03-09T21:00:00",
          27
        ],
        [
          "2026-03-09T22:00:00",
          23
        ],
        [
          "2026-03-09T23:00:00",
          19
        ],
        [
          "2026-03-10T00:00:00",
          12
        ],
        [
          "2026-03-10T01:00:00",
          8
        ],
        [
          "2026-03-10T02:00:00",
          4
        ],
        [
          "2026-03-10T03:00:00",
          4
        ],
        [
          "2026-03-10T04:00:00",
          9
        ],
        [
          "2026-03-10T05:00:00",
          12
        ],
        [
          "2026-03-10T06:00:00",
          20
        ],
        [
          "2026-03-10T07:00:00",
          28
        ],
        [
          "2026-03-10T08:00:00",
          35
        ],
        [
          "2026-03-10T09:00:00",
          36
        ],
        [
          "2026-03-10T10:00:00",
          39
        ],
        [
          "2026-03-10T11:00:00",
          36
        ],
        [
          "2026-03-10T12:00:00",
          34
        ],
        [
          "2026-03-10T13:00:00",
          26
        ],
        [
          "2026-03-10T14:00:00",
          22
        ],
        [
          "2026-03-10T15:00:00",
          19
        ],
        [
          "2026-03-10T16:00:00",
          21
        ],
        [
          "2026-03-10T17:00:00",
          22
        ],
        [
          "2026-03-10T18:00:00",
          23
        ],
        [
          "2026-03-10T19:00:00",
          25
        ],
        [
          "2026-03-10T20:00:00",
          28
        ],
        [
          "2026-03-10T21:00:00",
          27
        ],
        [
          "2026-03-10T22:00:00",
          23
        ],
        [
          "2026-03-10T23:00:00",
          17
        ],
        [
          "2026-03-11T00:00:00",
          11
        ],
        [
          "2026-03-11T01:00:00",
          9
        ],
        [
          "2026-03-11T02:00:00",
          4
        ],
        [
          "2026-03-11T03:00:00",
          3
        ],
        [
          "2026-03-11T04:00:00",
          6
        ],
        [
          "2026-03-11T05:00:00",
          11
        ],
        [
          "2026-03-11T06:00:00",
          22
        ],
        [
          "2026-03-11T07:00:00",
          29
        ],
        [
          "2026-03-11T08:00:00",
          33
        ],
        [
          "2026-03-11T09:00:00",
          38
        ],
        [
          "2026-03-11T10:00:00",
          38
        ],
        [
          "2026-03-11T11:00:00",
          37
        ],
        [
          "2026-03-11T12:00:00",
          32
        ],
        [
          "2026-03-11T13:00:00",
          26
        ],
        [
          "2026-03-11T14:00:00",
          24
        ],
        [
          "2026-03-11T15:00:00",
          21
        ],
        [
          "2026-03-11T16:00:00",
          21
        ],
        [
          "2026-03-11T17:00:00",
          20
        ],
        [
          "2026-03-11T18:00:00",
          24
        ],
        [
          "2026-03-11T19:00:00",
          27
        ],
        [
          "2026-03-11T20:00:00",
          28
        ],
        [
          "2026-03-11T21:00:00",
          27
        ],
        [
          "2026-03-11T22:00:00",
          23
        ],
        [
          "2026-03-11T23:00:00",
          18
        ],
        [
          "2026-03-12T00:00:00",
          11
        ],
        [
          "2026-03-12T01:00:00",
          8
        ],
        [
          "2026-03-12T02:00:00",
          3
        ],
        [
          "2026-03-12T03:00:00",
          4
        ],
        [
          "2026-03-12T04:00:00",
          7
        ],
        [
          "2026-03-12T05:00:00",
          13
        ],
        [
          "2026-03-12T06:00:00",
          19
        ],
        [
          "2026-03-12T07:00:00",
          29
        ],
        [
          "2026-03-12T08:00:00",
          34
        ],
        [
          "2026-03-12T09:00:00",
          41
        ],
        [
          "2026-03-12T10:00:00",
          38
        ],
        [
          "2026-03-12T11:00:00",
          37
        ],
        [
          "2026-03-12T12:00:00",
          31
        ],
        [
          "2026-03-12T13:00:00",
          29
        ],
        [
          "2026-03-12T14:00:00",
          26
        ],
        [
          "2026-03-12T15:00:00",
          21
        ],
        [
          "2026-03-12T16:00:00",
          19
        ],
        [
          "2026-03-12T17:00:00",
          21
        ],
        [
          "2026-03-12T18:00:00",
          22
        ],
        [
          "2026-03-12T19:00:00",
          25
        ],
        [
          "2026-03-12T20:00:00",
          26
        ],
        [
          "2026-03-12T21:00:00",
          25
        ],
        [
          "2026-03-12T22:00:00",
          21
        ],
        [
          "2026-03-12T23:00:00",
          17
        ],
        [
          "2026-03-13T00:00:00",
          15
        ],
        [
          "2026-03-13T01:00:00",
          6
        ],
        [
          "2026-03-13T02:00:00",
          4
        ],
        [
          "2026-03-13T03:00:00",
          6
        ],
        [
          "2026-03-13T04:00:00",
          7
        ],
        [
          "2026-03-13T05:00:00",
          13
        ],
        [
          "2026-03-13T06:00:00",
          21
        ],
        [
          "2026-03-13T07:00:00",
          27
        ],
        [
          "2026-03-13T08:00:00",
          33
        ],
        [
          "2026-03-13T09:00:00",
          38
        ],
        [
          "2026-03-13T10:00:00",
          38
        ],
        [
          "2026-03-13T11:00:00",
          36
        ],
        [
          "2026-03-13T12:00:00",
          33
        ],
        [
          "2026-03-13T13:00:00",
          25
        ],
        [
          "2026-03-13T14:00:00",
          23
        ],
        [
          "2026-03-13T15:00:00",
          21
        ],
        [
          "2026-03-13T16:00:00",
          20
        ],
        [
          "2026-03-13T17:00:00",
          23
        ],
        [
          "2026-03-13T18:00:00",
          23
        ],
        [
          "2026-03-13T19:00:00",
          26
        ],
        [
          "2026-03-13T20:00:00",
          28
        ],
        [
          "2026-03-13T21:00:00",
          25
        ],
        [
          "2026-03-13T22:00:00",
          21
        ],
        [
          "2026-03-13T23:00:00",
          17
        ],
        [
          "2026-03-14T00:00:00",
          12
        ],
        [
          "2026-03-14T01:00:00",
          7
        ],
        [
          "2026-03-14T02:00:00",
          4
        ],
        [
          "2026-03-14T03:00:00",
          4
        ],
        [
          "2026-03-14T04:00:00",
          7
        ],
        [
          "2026-03-14T05:00:00",
          11
        ],
        [
          "2026-03-14T06:00:00",
          21
        ],
        [
          "2026-03-14T07:00:00",
          29
        ],
        [
          "2026-03-14T08:00:00",
          33
        ],
        [
          "2026-03-14T09:00:00",
          38
        ],
        [
          "2026-03-14T10:00:00",
          39
        ],
        [
          "2026-03-14T11:00:00",
          37
        ],
        [
          "2026-03-14T12:00:00",
          31
        ],
        [
          "2026-03-14T13:00:00",
          26
        ],
        [
          "2026-03-14T14:00:00",
          24
        ],
        [
          "2026-03-14T15:00:00",
          20
        ],
        [
          "2026-03-14T16:00:00",
          20
        ],
        [
          "2026-03-14T17:00:00",
          22
        ],
        [
          "2026-03-14T18:00:00",
          24
        ],
        [
          "2026-03-14T19:00:00",
          26
        ],
        [
          "2026-03-14T20:00:00",
          26
        ],
        [
          "2026-03-14T21:00:00",
          27
        ],
        [
          "2026-03-14T22:00:00",
          23
        ],
        [
          "2026-03-14T23:00:00",
          18
        ],
        [
          "2026-03-15T00:00:00",
          13
        ],
        [
          "2026-03-15T01:00:00",
          10
        ],
        [
          "2026-03-15T02:00:00",
          4
        ],
        [
          "2026-03-15T03:00:00",
          3
        ],
        [
          "2026-03-15T04:00:00",
          7
        ],
        [
          "2026-03-15T05:00:00",
          14
        ],
        [
          "2026-03-15T06:00:00",
          20
        ],
        [
          "2026-03-15T07:00:00",
          29
        ],
        [
          "2026-03-15T08:00:00",
          34
        ],
        [
          "2026-03-15T09:00:00",
          39
        ],
        [
          "2026-03-15T10:00:00",
          38
        ],
        [
          "2026-03-15T11:00:00",
          37
        ],
        [
          "2026-03-15T12:00:00",
          32
        ],
        [
          "2026-03-15T13:00:00",
          26
        ],
        [
          "2026-03-15T14:00:00",
          23
        ],
        [
          "2026-03-15T15:00:00",
          21
        ]
      ]
    }
    </pre>
    </code>
    </details>

    <p>Now let me visualize it!</p>


    <details>
    <summary>Ran visualize_data</summary>
    <p><strong>Assistant sent this info to data-visualization</strong></p>
    <code>
    <pre class="tool">
    {
      "inputs": {
        "data": {
          "Product A - Units Sold": {
            "columns": [
              "timestamp",
              "units_sold"
            ],
            "rows": [
              [
                "2026-03-08T15:00:00",
                21
              ],
              [
                "2026-03-08T16:00:00",
                20
              ],
              [
                "2026-03-08T17:00:00",
                21
              ],
              [
                "2026-03-08T18:00:00",
                25
              ],
              [
                "2026-03-08T19:00:00",
                23
              ],
              [
                "2026-03-08T20:00:00",
                24
              ],
              [
                "2026-03-08T21:00:00",
                26
              ],
              [
                "2026-03-08T22:00:00",
                25
              ],
              [
                "2026-03-08T23:00:00",
                17
              ],
              [
                "2026-03-09T00:00:00",
                13
              ],
              [
                "2026-03-09T01:00:00",
                7
              ],
              [
                "2026-03-09T02:00:00",
                4
              ],
              [
                "2026-03-09T03:00:00",
                2
              ],
              [
                "2026-03-09T04:00:00",
                7
              ],
              [
                "2026-03-09T05:00:00",
                14
              ],
              [
                "2026-03-09T06:00:00",
                22
              ],
              [
                "2026-03-09T07:00:00",
                26
              ],
              [
                "2026-03-09T08:00:00",
                35
              ],
              [
                "2026-03-09T09:00:00",
                37
              ],
              [
                "2026-03-09T10:00:00",
                39
              ],
              [
                "2026-03-09T11:00:00",
                36
              ],
              [
                "2026-03-09T12:00:00",
                32
              ],
              [
                "2026-03-09T13:00:00",
                27
              ],
              [
                "2026-03-09T14:00:00",
                23
              ],
              [
                "2026-03-09T15:00:00",
                20
              ],
              [
                "2026-03-09T16:00:00",
                19
              ],
              [
                "2026-03-09T17:00:00",
                20
              ],
              [
                "2026-03-09T18:00:00",
                22
              ],
              [
                "2026-03-09T19:00:00",
                25
              ],
              [
                "2026-03-09T20:00:00",
                26
              ],
              [
                "2026-03-09T21:00:00",
                27
              ],
              [
                "2026-03-09T22:00:00",
                23
              ],
              [
                "2026-03-09T23:00:00",
                19
              ],
              [
                "2026-03-10T00:00:00",
                12
              ],
              [
                "2026-03-10T01:00:00",
                8
              ],
              [
                "2026-03-10T02:00:00",
                4
              ],
              [
                "2026-03-10T03:00:00",
                4
              ],
              [
                "2026-03-10T04:00:00",
                9
              ],
              [
                "2026-03-10T05:00:00",
                12
              ],
              [
                "2026-03-10T06:00:00",
                20
              ],
              [
                "2026-03-10T07:00:00",
                28
              ],
              [
                "2026-03-10T08:00:00",
                35
              ],
              [
                "2026-03-10T09:00:00",
                36
              ],
              [
                "2026-03-10T10:00:00",
                39
              ],
              [
                "2026-03-10T11:00:00",
                36
              ],
              [
                "2026-03-10T12:00:00",
                34
              ],
              [
                "2026-03-10T13:00:00",
                26
              ],
              [
                "2026-03-10T14:00:00",
                22
              ],
              [
                "2026-03-10T15:00:00",
                19
              ],
              [
                "2026-03-10T16:00:00",
                21
              ],
              [
                "2026-03-10T17:00:00",
                22
              ],
              [
                "2026-03-10T18:00:00",
                23
              ],
              [
                "2026-03-10T19:00:00",
                25
              ],
              [
                "2026-03-10T20:00:00",
                28
              ],
              [
                "2026-03-10T21:00:00",
                27
              ],
              [
                "2026-03-10T22:00:00",
                23
              ],
              [
                "2026-03-10T23:00:00",
                17
              ],
              [
                "2026-03-11T00:00:00",
                11
              ],
              [
                "2026-03-11T01:00:00",
                9
              ],
              [
                "2026-03-11T02:00:00",
                4
              ],
              [
                "2026-03-11T03:00:00",
                3
              ],
              [
                "2026-03-11T04:00:00",
                6
              ],
              [
                "2026-03-11T05:00:00",
                11
              ],
              [
                "2026-03-11T06:00:00",
                22
              ],
              [
                "2026-03-11T07:00:00",
                29
              ],
              [
                "2026-03-11T08:00:00",
                33
              ],
              [
                "2026-03-11T09:00:00",
                38
              ],
              [
                "2026-03-11T10:00:00",
                38
              ],
              [
                "2026-03-11T11:00:00",
                37
              ],
              [
                "2026-03-11T12:00:00",
                32
              ],
              [
                "2026-03-11T13:00:00",
                26
              ],
              [
                "2026-03-11T14:00:00",
                24
              ],
              [
                "2026-03-11T15:00:00",
                21
              ],
              [
                "2026-03-11T16:00:00",
                21
              ],
              [
                "2026-03-11T17:00:00",
                20
              ],
              [
                "2026-03-11T18:00:00",
                24
              ],
              [
                "2026-03-11T19:00:00",
                27
              ],
              [
                "2026-03-11T20:00:00",
                28
              ],
              [
                "2026-03-11T21:00:00",
                27
              ],
              [
                "2026-03-11T22:00:00",
                23
              ],
              [
                "2026-03-11T23:00:00",
                18
              ],
              [
                "2026-03-12T00:00:00",
                11
              ],
              [
                "2026-03-12T01:00:00",
                8
              ],
              [
                "2026-03-12T02:00:00",
                3
              ],
              [
                "2026-03-12T03:00:00",
                4
              ],
              [
                "2026-03-12T04:00:00",
                7
              ],
              [
                "2026-03-12T05:00:00",
                13
              ],
              [
                "2026-03-12T06:00:00",
                19
              ],
              [
                "2026-03-12T07:00:00",
                29
              ],
              [
                "2026-03-12T08:00:00",
                34
              ],
              [
                "2026-03-12T09:00:00",
                41
              ],
              [
                "2026-03-12T10:00:00",
                38
              ],
              [
                "2026-03-12T11:00:00",
                37
              ],
              [
                "2026-03-12T12:00:00",
                31
              ],
              [
                "2026-03-12T13:00:00",
                29
              ],
              [
                "2026-03-12T14:00:00",
                26
              ],
              [
                "2026-03-12T15:00:00",
                21
              ],
              [
                "2026-03-12T16:00:00",
                19
              ],
              [
                "2026-03-12T17:00:00",
                21
              ],
              [
                "2026-03-12T18:00:00",
                22
              ],
              [
                "2026-03-12T19:00:00",
                25
              ],
              [
                "2026-03-12T20:00:00",
                26
              ],
              [
                "2026-03-12T21:00:00",
                25
              ],
              [
                "2026-03-12T22:00:00",
                21
              ],
              [
                "2026-03-12T23:00:00",
                17
              ],
              [
                "2026-03-13T00:00:00",
                15
              ],
              [
                "2026-03-13T01:00:00",
                6
              ],
              [
                "2026-03-13T02:00:00",
                4
              ],
              [
                "2026-03-13T03:00:00",
                6
              ],
              [
                "2026-03-13T04:00:00",
                7
              ],
              [
                "2026-03-13T05:00:00",
                13
              ],
              [
                "2026-03-13T06:00:00",
                21
              ],
              [
                "2026-03-13T07:00:00",
                27
              ],
              [
                "2026-03-13T08:00:00",
                33
              ],
              [
                "2026-03-13T09:00:00",
                38
              ],
              [
                "2026-03-13T10:00:00",
                38
              ],
              [
                "2026-03-13T11:00:00",
                36
              ],
              [
                "2026-03-13T12:00:00",
                33
              ],
              [
                "2026-03-13T13:00:00",
                25
              ],
              [
                "2026-03-13T14:00:00",
                23
              ],
              [
                "2026-03-13T15:00:00",
                21
              ],
              [
                "2026-03-13T16:00:00",
                20
              ],
              [
                "2026-03-13T17:00:00",
                23
              ],
              [
                "2026-03-13T18:00:00",
                23
              ],
              [
                "2026-03-13T19:00:00",
                26
              ],
              [
                "2026-03-13T20:00:00",
                28
              ],
              [
                "2026-03-13T21:00:00",
                25
              ],
              [
                "2026-03-13T22:00:00",
                21
              ],
              [
                "2026-03-13T23:00:00",
                17
              ],
              [
                "2026-03-14T00:00:00",
                12
              ],
              [
                "2026-03-14T01:00:00",
                7
              ],
              [
                "2026-03-14T02:00:00",
                4
              ],
              [
                "2026-03-14T03:00:00",
                4
              ],
              [
                "2026-03-14T04:00:00",
                7
              ],
              [
                "2026-03-14T05:00:00",
                11
              ],
              [
                "2026-03-14T06:00:00",
                21
              ],
              [
                "2026-03-14T07:00:00",
                29
              ],
              [
                "2026-03-14T08:00:00",
                33
              ],
              [
                "2026-03-14T09:00:00",
                38
              ],
              [
                "2026-03-14T10:00:00",
                39
              ],
              [
                "2026-03-14T11:00:00",
                37
              ],
              [
                "2026-03-14T12:00:00",
                31
              ],
              [
                "2026-03-14T13:00:00",
                26
              ],
              [
                "2026-03-14T14:00:00",
                24
              ],
              [
                "2026-03-14T15:00:00",
                20
              ],
              [
                "2026-03-14T16:00:00",
                20
              ],
              [
                "2026-03-14T17:00:00",
                22
              ],
              [
                "2026-03-14T18:00:00",
                24
              ],
              [
                "2026-03-14T19:00:00",
                26
              ],
              [
                "2026-03-14T20:00:00",
                26
              ],
              [
                "2026-03-14T21:00:00",
                27
              ],
              [
                "2026-03-14T22:00:00",
                23
              ],
              [
                "2026-03-14T23:00:00",
                18
              ],
              [
                "2026-03-15T00:00:00",
                13
              ],
              [
                "2026-03-15T01:00:00",
                10
              ],
              [
                "2026-03-15T02:00:00",
                4
              ],
              [
                "2026-03-15T03:00:00",
                3
              ],
              [
                "2026-03-15T04:00:00",
                7
              ],
              [
                "2026-03-15T05:00:00",
                14
              ],
              [
                "2026-03-15T06:00:00",
                20
              ],
              [
                "2026-03-15T07:00:00",
                29
              ],
              [
                "2026-03-15T08:00:00",
                34
              ],
              [
                "2026-03-15T09:00:00",
                39
              ],
              [
                "2026-03-15T10:00:00",
                38
              ],
              [
                "2026-03-15T11:00:00",
                37
              ],
              [
                "2026-03-15T12:00:00",
                32
              ],
              [
                "2026-03-15T13:00:00",
                26
              ],
              [
                "2026-03-15T14:00:00",
                23
              ],
              [
                "2026-03-15T15:00:00",
                21
              ]
            ]
          }
        }
      }
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "columns": [
        "product_id",
        "product_name"
      ],
      "rows": [
        [
          1,
          "Product A"
        ],
        [
          2,
          "Product B"
        ],
        [
          3,
          "Product C"
        ]
      ]
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    http://localhost:8004/plot_28706dcf80a5404fa8cf6cdc017c8152.html
    </pre>
    </code>
    </details>

    <div style="display: block; margin-right: 1rem">
    <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.4.0.min.js" integrity="sha256-KEmPoupLpFyGMyGAiOsiNDbKDKAvxXAn/W+oQa0ZAfk=" crossorigin="anonymous"></script>                <div id="db7d5039-36e5-47d7-91ae-687797db871f" class="plotly-graph-div" style="height:250px; width:100%;"></div>            <script>                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("db7d5039-36e5-47d7-91ae-687797db871f")) {                    Plotly.newPlot(                        "db7d5039-36e5-47d7-91ae-687797db871f",                        [{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#838383","width":1},"mode":"lines","name":"Historical Data","showlegend":true,"x":["2026-03-08T15:00:00","2026-03-08T16:00:00","2026-03-08T17:00:00","2026-03-08T18:00:00","2026-03-08T19:00:00","2026-03-08T20:00:00","2026-03-08T21:00:00","2026-03-08T22:00:00","2026-03-08T23:00:00","2026-03-09T00:00:00","2026-03-09T01:00:00","2026-03-09T02:00:00","2026-03-09T03:00:00","2026-03-09T04:00:00","2026-03-09T05:00:00","2026-03-09T06:00:00","2026-03-09T07:00:00","2026-03-09T08:00:00","2026-03-09T09:00:00","2026-03-09T10:00:00","2026-03-09T11:00:00","2026-03-09T12:00:00","2026-03-09T13:00:00","2026-03-09T14:00:00","2026-03-09T15:00:00","2026-03-09T16:00:00","2026-03-09T17:00:00","2026-03-09T18:00:00","2026-03-09T19:00:00","2026-03-09T20:00:00","2026-03-09T21:00:00","2026-03-09T22:00:00","2026-03-09T23:00:00","2026-03-10T00:00:00","2026-03-10T01:00:00","2026-03-10T02:00:00","2026-03-10T03:00:00","2026-03-10T04:00:00","2026-03-10T05:00:00","2026-03-10T06:00:00","2026-03-10T07:00:00","2026-03-10T08:00:00","2026-03-10T09:00:00","2026-03-10T10:00:00","2026-03-10T11:00:00","2026-03-10T12:00:00","2026-03-10T13:00:00","2026-03-10T14:00:00","2026-03-10T15:00:00","2026-03-10T16:00:00","2026-03-10T17:00:00","2026-03-10T18:00:00","2026-03-10T19:00:00","2026-03-10T20:00:00","2026-03-10T21:00:00","2026-03-10T22:00:00","2026-03-10T23:00:00","2026-03-11T00:00:00","2026-03-11T01:00:00","2026-03-11T02:00:00","2026-03-11T03:00:00","2026-03-11T04:00:00","2026-03-11T05:00:00","2026-03-11T06:00:00","2026-03-11T07:00:00","2026-03-11T08:00:00","2026-03-11T09:00:00","2026-03-11T10:00:00","2026-03-11T11:00:00","2026-03-11T12:00:00","2026-03-11T13:00:00","2026-03-11T14:00:00","2026-03-11T15:00:00","2026-03-11T16:00:00","2026-03-11T17:00:00","2026-03-11T18:00:00","2026-03-11T19:00:00","2026-03-11T20:00:00","2026-03-11T21:00:00","2026-03-11T22:00:00","2026-03-11T23:00:00","2026-03-12T00:00:00","2026-03-12T01:00:00","2026-03-12T02:00:00","2026-03-12T03:00:00","2026-03-12T04:00:00","2026-03-12T05:00:00","2026-03-12T06:00:00","2026-03-12T07:00:00","2026-03-12T08:00:00","2026-03-12T09:00:00","2026-03-12T10:00:00","2026-03-12T11:00:00","2026-03-12T12:00:00","2026-03-12T13:00:00","2026-03-12T14:00:00","2026-03-12T15:00:00","2026-03-12T16:00:00","2026-03-12T17:00:00","2026-03-12T18:00:00","2026-03-12T19:00:00","2026-03-12T20:00:00","2026-03-12T21:00:00","2026-03-12T22:00:00","2026-03-12T23:00:00","2026-03-13T00:00:00","2026-03-13T01:00:00","2026-03-13T02:00:00","2026-03-13T03:00:00","2026-03-13T04:00:00","2026-03-13T05:00:00","2026-03-13T06:00:00","2026-03-13T07:00:00","2026-03-13T08:00:00","2026-03-13T09:00:00","2026-03-13T10:00:00","2026-03-13T11:00:00","2026-03-13T12:00:00","2026-03-13T13:00:00","2026-03-13T14:00:00","2026-03-13T15:00:00","2026-03-13T16:00:00","2026-03-13T17:00:00","2026-03-13T18:00:00","2026-03-13T19:00:00","2026-03-13T20:00:00","2026-03-13T21:00:00","2026-03-13T22:00:00","2026-03-13T23:00:00","2026-03-14T00:00:00","2026-03-14T01:00:00","2026-03-14T02:00:00","2026-03-14T03:00:00","2026-03-14T04:00:00","2026-03-14T05:00:00","2026-03-14T06:00:00","2026-03-14T07:00:00","2026-03-14T08:00:00","2026-03-14T09:00:00","2026-03-14T10:00:00","2026-03-14T11:00:00","2026-03-14T12:00:00","2026-03-14T13:00:00","2026-03-14T14:00:00","2026-03-14T15:00:00","2026-03-14T16:00:00","2026-03-14T17:00:00","2026-03-14T18:00:00","2026-03-14T19:00:00","2026-03-14T20:00:00","2026-03-14T21:00:00","2026-03-14T22:00:00","2026-03-14T23:00:00","2026-03-15T00:00:00","2026-03-15T01:00:00","2026-03-15T02:00:00","2026-03-15T03:00:00","2026-03-15T04:00:00","2026-03-15T05:00:00","2026-03-15T06:00:00","2026-03-15T07:00:00","2026-03-15T08:00:00","2026-03-15T09:00:00","2026-03-15T10:00:00","2026-03-15T11:00:00","2026-03-15T12:00:00","2026-03-15T13:00:00","2026-03-15T14:00:00","2026-03-15T15:00:00"],"y":[21,20,21,25,23,24,26,25,17,13,7,4,2,7,14,22,26,35,37,39,36,32,27,23,20,19,20,22,25,26,27,23,19,12,8,4,4,9,12,20,28,35,36,39,36,34,26,22,19,21,22,23,25,28,27,23,17,11,9,4,3,6,11,22,29,33,38,38,37,32,26,24,21,21,20,24,27,28,27,23,18,11,8,3,4,7,13,19,29,34,41,38,37,31,29,26,21,19,21,22,25,26,25,21,17,15,6,4,6,7,13,21,27,33,38,38,36,33,25,23,21,20,23,23,26,28,25,21,17,12,7,4,4,7,11,21,29,33,38,39,37,31,26,24,20,20,22,24,26,26,27,23,18,13,10,4,3,7,14,20,29,34,39,38,37,32,26,23,21],"type":"scatter","xaxis":"x","yaxis":"y"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"tickfont":{"color":"#24292f","size":10},"type":"date","tickformat":"%b %d\u003cbr\u003e%Y","tickangle":0,"mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"yaxis":{"anchor":"x","domain":[0.0,1.0],"tickfont":{"color":"#24292f","size":10},"tickformat":",.0f","mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"annotations":[{"font":{"size":14,"color":"#24292f"},"showarrow":false,"text":"Product A - Units Sold","x":0.5,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"margin":{"t":80,"b":50,"l":50,"r":50},"hoverlabel":{"namelength":-1},"legend":{"font":{"color":"#24292f","size":12}, "x": 0, "y": 1.5, "orientation": "h"},"height":250,"paper_bgcolor":"white","plot_bgcolor":"white"},                        {"responsive": true}                    )                };</script>
    </div>

    <p>Here's the hourly sales chart for <strong>Product A</strong> over the most recent week (Mar 8–15, 2026). A few observations:</p>
    <ul>
    <li>📉 <strong>Daily trough:</strong> Sales consistently drop to very low values (2–4 units) in the <strong>early morning hours (2:00–4:00 AM)</strong></li>
    <li>📈 <strong>Daily peak:</strong> Sales peak around <strong>9:00–10:00 AM</strong>, reaching up to ~38–41 units</li>
    <li>🔄 <strong>Strong daily seasonality:</strong> The pattern repeats very consistently each day, suggesting a clear intraday cycle</li>
    <li>📊 <strong>Stable trend:</strong> There's no obvious upward or downward trend over the week — sales levels look steady</li>
    </ul>
    <p>Would you like me to <strong>forecast</strong> the next few days for Product A, or explore the other products as well?</p>
    </div>
    </div>

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Can we now use this data as input to forecast the subsequent 2 days?
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Assistant</div>
    <div class="message">
    <details>
    <summary>Thoughts</summary>
    <pre class="thought">
    The user wants to forecast the next 2 days (48 hours) for Product A using the data we already queried. The data is hourly, so 2 days = 48 time steps.

    I need to:
    1. Call generate_forecasts with the historical data as target, prediction_length=48, and some quantile levels
    2. Then visualize the results with the forecast overlaid

    Let me prepare the target array from the query results we already have.

    The rows are hourly data from 2026-03-08T15:00:00 to 2026-03-15T15:00:00.

    Let me extract the values:
    [21,20,21,25,23,24,26,25,17,13,7,4,2,7,14,22,26,35,37,39,36,32,27,23,20,19,20,22,25,26,27,23,19,12,8,4,4,9,12,20,28,35,36,39,36,34,26,22,19,21,22,23,25,28,27,23,17,11,9,4,3,6,11,22,29,33,38,38,37,32,26,24,21,21,20,24,27,28,27,23,18,11,8,3,4,7,13,19,29,34,41,38,37,31,29,26,21,19,21,22,25,26,25,21,17,15,6,4,6,7,13,21,27,33,38,38,36,33,25,23,21,20,23,23,26,28]

    Now I'm setting up the forecast parameters with the last observation at 2026-03-15T15:00:00, which means the 48-hour forecast will run from 2026-03-15T16:00:00 through 2026-03-17T15:00:00. I'm using quantile levels at 0.1, 0.5, and 0.9 to generate an 80% prediction interval.
    </pre>
    </details>

    <p>I'll use the hourly data we already retrieved as input to Chronos and forecast the next 48 hours (2 days)!</p>

    <details>
    <summary>Ran generate_forecasts</summary>
    <p><strong>Assistant sent this info to chronos-forecasting</strong></p>
    <code>
    <pre class="tool">
    {
      "target": [
        21,
        20,
        21,
        25,
        23,
        24,
        26,
        25,
        17,
        13,
        7,
        4,
        2,
        7,
        14,
        22,
        26,
        35,
        37,
        39,
        36,
        32,
        27,
        23,
        20,
        19,
        20,
        22,
        25,
        26,
        27,
        23,
        19,
        12,
        8,
        4,
        4,
        9,
        12,
        20,
        28,
        35,
        36,
        39,
        36,
        34,
        26,
        22,
        19,
        21,
        22,
        23,
        25,
        28,
        27,
        23,
        17,
        11,
        9,
        4,
        3,
        6,
        11,
        22,
        29,
        33,
        38,
        38,
        37,
        32,
        26,
        24,
        21,
        21,
        20,
        24,
        27,
        28,
        27,
        23,
        18,
        11,
        8,
        3,
        4,
        7,
        13,
        19,
        29,
        34,
        41,
        38,
        37,
        31,
        29,
        26,
        21,
        19,
        21,
        22,
        25,
        26,
        25,
        21,
        17,
        15,
        6,
        4,
        6,
        7,
        13,
        21,
        27,
        33,
        38,
        38,
        36,
        33,
        25,
        23,
        21,
        20,
        23,
        23,
        26,
        28,
        25,
        21,
        17,
        12,
        7,
        4,
        4,
        7,
        11,
        21,
        29,
        33,
        38,
        39,
        37,
        31,
        26,
        24,
        20,
        20,
        22,
        24,
        26,
        26,
        27,
        23,
        18,
        13,
        10,
        4,
        3,
        7,
        14,
        20,
        29,
        34,
        39,
        38,
        37,
        32,
        26,
        23,
        21
      ],
      "prediction_length": 48,
      "quantile_levels": [
        0.1,
        0.5,
        0.9
      ]
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "mean": [
        20.504526138305664,
        21.805526733398438,
        23.69278907775879,
        25.529420852661133,
        26.827110290527344,
        26.324140548706055,
        22.897764205932617,
        17.852794647216797,
        12.321682929992676,
        7.192910194396973,
        3.9130783081054688,
        3.9800643920898438,
        7.332183837890625,
        12.986684799194336,
        20.668590545654297,
        27.952547073364258,
        33.47609329223633,
        38.25192642211914,
        38.400447845458984,
        36.45668029785156,
        31.858444213867188,
        26.570825576782227,
        23.2839412689209,
        20.51091957092285,
        20.30457305908203,
        21.678436279296875,
        23.642099380493164,
        25.584678649902344,
        26.985809326171875,
        26.484193801879883,
        23.02522850036621,
        18.039037704467773,
        12.611462593078613,
        7.602766990661621,
        4.303823471069336,
        4.018426895141602,
        7.423919677734375,
        13.13841438293457,
        20.692811965942383,
        28.037954330444336,
        33.97454071044922,
        38.231040954589844,
        38.78217697143555,
        36.92472457885742,
        32.19001388549805,
        26.802392959594727,
        23.54244041442871,
        20.698514938354492
      ],
      "0.1": [
        19.15822410583496,
        20.50179100036621,
        22.401607513427734,
        24.21921157836914,
        25.45020294189453,
        24.837438583374023,
        21.506221771240234,
        16.50567626953125,
        10.83835220336914,
        5.582372665405273,
        2.285032272338867,
        2.251415252685547,
        5.641073226928711,
        11.404353141784668,
        18.795143127441406,
        25.81307601928711,
        31.14609718322754,
        35.49599838256836,
        35.904144287109375,
        34.05424499511719,
        29.796527862548828,
        24.967273712158203,
        21.782663345336914,
        19.394075393676758,
        19.236570358276367,
        20.604127883911133,
        22.458040237426758,
        24.36791229248047,
        25.4338436126709,
        24.74117088317871,
        21.389162063598633,
        16.495697021484375,
        10.960190773010254,
        5.869045257568359,
        2.5734004974365234,
        2.361164093017578,
        5.705011367797852,
        11.369453430175781,
        18.547435760498047,
        25.60714340209961,
        31.265169143676758,
        35.313446044921875,
        36.011741638183594,
        34.2758674621582,
        29.958742141723633,
        25.103273391723633,
        21.822275161743164,
        19.32248306274414
      ],
      "0.5": [
        20.504526138305664,
        21.805526733398438,
        23.69278907775879,
        25.529420852661133,
        26.827110290527344,
        26.324140548706055,
        22.897764205932617,
        17.852794647216797,
        12.321682929992676,
        7.192910194396973,
        3.9130783081054688,
        3.9800643920898438,
        7.332183837890625,
        12.986684799194336,
        20.668590545654297,
        27.952547073364258,
        33.47609329223633,
        38.25192642211914,
        38.400447845458984,
        36.45668029785156,
        31.858444213867188,
        26.570825576782227,
        23.2839412689209,
        20.51091957092285,
        20.30457305908203,
        21.678436279296875,
        23.642099380493164,
        25.584678649902344,
        26.985809326171875,
        26.484193801879883,
        23.02522850036621,
        18.039037704467773,
        12.611462593078613,
        7.602766990661621,
        4.303823471069336,
        4.018426895141602,
        7.423919677734375,
        13.13841438293457,
        20.692811965942383,
        28.037954330444336,
        33.97454071044922,
        38.231040954589844,
        38.78217697143555,
        36.92472457885742,
        32.19001388549805,
        26.802392959594727,
        23.54244041442871,
        20.698514938354492
      ],
      "0.9": [
        21.669395446777344,
        23.053070068359375,
        24.99215316772461,
        26.91657066345215,
        28.313753128051758,
        27.791156768798828,
        24.152969360351562,
        19.0040283203125,
        13.467171669006348,
        8.364714622497559,
        4.990467071533203,
        5.050144195556641,
        8.647828102111816,
        14.336263656616211,
        22.190526962280273,
        29.632858276367188,
        35.39604949951172,
        40.310279846191406,
        40.52391815185547,
        38.57139587402344,
        33.898765563964844,
        28.34296226501465,
        25.04033088684082,
        22.027423858642578,
        21.667360305786133,
        22.99347686767578,
        24.871681213378906,
        26.878807067871094,
        28.497577667236328,
        28.168556213378906,
        24.56938934326172,
        19.346378326416016,
        14.037256240844727,
        8.928860664367676,
        5.4414215087890625,
        5.182031631469727,
        8.82529354095459,
        14.672514915466309,
        22.400375366210938,
        29.78118896484375,
        35.815494537353516,
        40.36121368408203,
        40.88827133178711,
        39.124969482421875,
        34.39842224121094,
        28.938234329223633,
        25.735769271850586,
        22.512235641479492
      ]
    }
    </pre>
    </code>
    </details>

    <p>Got the forecasts! Now let me visualize the historical data alongside the forecast.</p>

    <div style="display: block; margin-right: 1rem">
    <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.4.0.min.js" integrity="sha256-KEmPoupLpFyGMyGAiOsiNDbKDKAvxXAn/W+oQa0ZAfk=" crossorigin="anonymous"></script>                <div id="7bfbe82a-aa39-4365-a5ac-e074ea050e62" class="plotly-graph-div" style="height:250px; width:100%;"></div>            <script>                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("7bfbe82a-aa39-4365-a5ac-e074ea050e62")) {                    Plotly.newPlot(                        "7bfbe82a-aa39-4365-a5ac-e074ea050e62",                        [{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q90%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[21.67,23.05,24.99,26.92,28.31,27.79,24.15,19,13.47,8.36,4.99,5.05,8.65,14.34,22.19,29.63,35.4,40.31,40.52,38.57,33.9,28.34,25.04,22.03,21.67,23,24.87,26.88,28.5,28.17,24.57,19.35,14.04,8.93,5.44,5.18,8.83,14.67,22.4,29.78,35.82,40.36,40.89,39.12,34.4,28.94,25.74,22.51],"type":"scatter","xaxis":"x","yaxis":"y"},{"fill":"tonexty","fillcolor":"#c2e5ff","hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q10%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[19.16,20.5,22.4,24.22,25.45,24.84,21.51,16.51,10.84,5.58,2.29,2.25,5.64,11.4,18.8,25.81,31.15,35.5,35.9,34.05,29.8,24.97,21.78,19.39,19.24,20.6,22.46,24.37,25.43,24.74,21.39,16.5,10.96,5.87,2.57,2.36,5.71,11.37,18.55,25.61,31.27,35.31,36.01,34.28,29.96,25.1,21.82,19.32],"type":"scatter","xaxis":"x","yaxis":"y"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#0588f0","dash":"dot","width":1},"mode":"lines","name":"Predicted Median","showlegend":true,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[20.5,21.81,23.69,25.53,26.83,26.32,22.9,17.85,12.32,7.19,3.91,3.98,7.33,12.99,20.67,27.95,33.48,38.25,38.4,36.46,31.86,26.57,23.28,20.51,20.3,21.68,23.64,25.58,26.99,26.48,23.03,18.04,12.61,7.6,4.3,4.02,7.42,13.14,20.69,28.04,33.97,38.23,38.78,36.92,32.19,26.8,23.54,20.7],"type":"scatter","xaxis":"x","yaxis":"y"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#838383","width":1},"mode":"lines","name":"Historical Data","showlegend":true,"x":["2026-03-08T15:00:00","2026-03-08T16:00:00","2026-03-08T17:00:00","2026-03-08T18:00:00","2026-03-08T19:00:00","2026-03-08T20:00:00","2026-03-08T21:00:00","2026-03-08T22:00:00","2026-03-08T23:00:00","2026-03-09T00:00:00","2026-03-09T01:00:00","2026-03-09T02:00:00","2026-03-09T03:00:00","2026-03-09T04:00:00","2026-03-09T05:00:00","2026-03-09T06:00:00","2026-03-09T07:00:00","2026-03-09T08:00:00","2026-03-09T09:00:00","2026-03-09T10:00:00","2026-03-09T11:00:00","2026-03-09T12:00:00","2026-03-09T13:00:00","2026-03-09T14:00:00","2026-03-09T15:00:00","2026-03-09T16:00:00","2026-03-09T17:00:00","2026-03-09T18:00:00","2026-03-09T19:00:00","2026-03-09T20:00:00","2026-03-09T21:00:00","2026-03-09T22:00:00","2026-03-09T23:00:00","2026-03-10T00:00:00","2026-03-10T01:00:00","2026-03-10T02:00:00","2026-03-10T03:00:00","2026-03-10T04:00:00","2026-03-10T05:00:00","2026-03-10T06:00:00","2026-03-10T07:00:00","2026-03-10T08:00:00","2026-03-10T09:00:00","2026-03-10T10:00:00","2026-03-10T11:00:00","2026-03-10T12:00:00","2026-03-10T13:00:00","2026-03-10T14:00:00","2026-03-10T15:00:00","2026-03-10T16:00:00","2026-03-10T17:00:00","2026-03-10T18:00:00","2026-03-10T19:00:00","2026-03-10T20:00:00","2026-03-10T21:00:00","2026-03-10T22:00:00","2026-03-10T23:00:00","2026-03-11T00:00:00","2026-03-11T01:00:00","2026-03-11T02:00:00","2026-03-11T03:00:00","2026-03-11T04:00:00","2026-03-11T05:00:00","2026-03-11T06:00:00","2026-03-11T07:00:00","2026-03-11T08:00:00","2026-03-11T09:00:00","2026-03-11T10:00:00","2026-03-11T11:00:00","2026-03-11T12:00:00","2026-03-11T13:00:00","2026-03-11T14:00:00","2026-03-11T15:00:00","2026-03-11T16:00:00","2026-03-11T17:00:00","2026-03-11T18:00:00","2026-03-11T19:00:00","2026-03-11T20:00:00","2026-03-11T21:00:00","2026-03-11T22:00:00","2026-03-11T23:00:00","2026-03-12T00:00:00","2026-03-12T01:00:00","2026-03-12T02:00:00","2026-03-12T03:00:00","2026-03-12T04:00:00","2026-03-12T05:00:00","2026-03-12T06:00:00","2026-03-12T07:00:00","2026-03-12T08:00:00","2026-03-12T09:00:00","2026-03-12T10:00:00","2026-03-12T11:00:00","2026-03-12T12:00:00","2026-03-12T13:00:00","2026-03-12T14:00:00","2026-03-12T15:00:00","2026-03-12T16:00:00","2026-03-12T17:00:00","2026-03-12T18:00:00","2026-03-12T19:00:00","2026-03-12T20:00:00","2026-03-12T21:00:00","2026-03-12T22:00:00","2026-03-12T23:00:00","2026-03-13T00:00:00","2026-03-13T01:00:00","2026-03-13T02:00:00","2026-03-13T03:00:00","2026-03-13T04:00:00","2026-03-13T05:00:00","2026-03-13T06:00:00","2026-03-13T07:00:00","2026-03-13T08:00:00","2026-03-13T09:00:00","2026-03-13T10:00:00","2026-03-13T11:00:00","2026-03-13T12:00:00","2026-03-13T13:00:00","2026-03-13T14:00:00","2026-03-13T15:00:00","2026-03-13T16:00:00","2026-03-13T17:00:00","2026-03-13T18:00:00","2026-03-13T19:00:00","2026-03-13T20:00:00","2026-03-13T21:00:00","2026-03-13T22:00:00","2026-03-13T23:00:00","2026-03-14T00:00:00","2026-03-14T01:00:00","2026-03-14T02:00:00","2026-03-14T03:00:00","2026-03-14T04:00:00","2026-03-14T05:00:00","2026-03-14T06:00:00","2026-03-14T07:00:00","2026-03-14T08:00:00","2026-03-14T09:00:00","2026-03-14T10:00:00","2026-03-14T11:00:00","2026-03-14T12:00:00","2026-03-14T13:00:00","2026-03-14T14:00:00","2026-03-14T15:00:00","2026-03-14T16:00:00","2026-03-14T17:00:00","2026-03-14T18:00:00","2026-03-14T19:00:00","2026-03-14T20:00:00","2026-03-14T21:00:00","2026-03-14T22:00:00","2026-03-14T23:00:00","2026-03-15T00:00:00","2026-03-15T01:00:00","2026-03-15T02:00:00","2026-03-15T03:00:00","2026-03-15T04:00:00","2026-03-15T05:00:00","2026-03-15T06:00:00","2026-03-15T07:00:00","2026-03-15T08:00:00","2026-03-15T09:00:00","2026-03-15T10:00:00","2026-03-15T11:00:00","2026-03-15T12:00:00","2026-03-15T13:00:00","2026-03-15T14:00:00","2026-03-15T15:00:00"],"y":[21,20,21,25,23,24,26,25,17,13,7,4,2,7,14,22,26,35,37,39,36,32,27,23,20,19,20,22,25,26,27,23,19,12,8,4,4,9,12,20,28,35,36,39,36,34,26,22,19,21,22,23,25,28,27,23,17,11,9,4,3,6,11,22,29,33,38,38,37,32,26,24,21,21,20,24,27,28,27,23,18,11,8,3,4,7,13,19,29,34,41,38,37,31,29,26,21,19,21,22,25,26,25,21,17,15,6,4,6,7,13,21,27,33,38,38,36,33,25,23,21,20,23,23,26,28,25,21,17,12,7,4,4,7,11,21,29,33,38,39,37,31,26,24,20,20,22,24,26,26,27,23,18,13,10,4,3,7,14,20,29,34,39,38,37,32,26,23,21],"type":"scatter","xaxis":"x","yaxis":"y"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"tickfont":{"color":"#24292f","size":10},"type":"date","tickformat":"%b %d\u003cbr\u003e%Y","tickangle":0,"mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"yaxis":{"anchor":"x","domain":[0.0,1.0],"tickfont":{"color":"#24292f","size":10},"tickformat":",.0f","mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"annotations":[{"font":{"size":14,"color":"#24292f"},"showarrow":false,"text":"Product A","x":0.5,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"margin":{"t":80,"b":50,"l":50,"r":50},"hoverlabel":{"namelength":-1},"legend":{"font":{"color":"#24292f","size":12}, "x": 0, "y": 1.5, "orientation": "h"},"height":250,"paper_bgcolor":"white","plot_bgcolor":"white"},                        {"responsive": true}                    )                };            </script>
    </div>

    <p>Here's the forecast for <strong>Product A</strong> for the next 2 days (Mar 15, 16:00 → Mar 17, 15:00), with an <strong>80% prediction interval</strong> (shaded band between the 10th and 90th percentiles). Key takeaways:</p>
    <div style="margin-right:1rem">
    <ul>
    <li>🔄 <strong>Strong daily cycle preserved:</strong> Chronos has clearly picked up on the intraday seasonality — predicting low sales overnight (2–4 units) and peaks mid-morning (38–41 units)</li>
    <li>📏 <strong>Tight uncertainty bands:</strong> The prediction interval is relatively narrow, reflecting the high regularity and consistency of the historical pattern</li>
    <li>📈 <strong>Two full daily cycles forecasted:</strong> Both Mar 16 and Mar 17 show the same characteristic shape as past days</li>
    </ul>
    </div>
    <p>Would you like to do the same for <strong>Product B</strong> or <strong>Product C</strong>, or extend the forecast horizon further?</p>
    </div>
    </div>

    <div class="user-container">
    <div class="user-name">User</div>
    <div class="message">
    <p>
    Can we repeat the same analysis for the remaining products and show all the results in one plot?
    </p>
    </div>
    </div>

    <div class="ai-container">
    <div class="ai-name">Assistant</div>
    <div class="message">
    <details>
    <summary>Thoughts</summary>
    <pre class="thought">
    The user wants to repeat the same analysis for Products B and C, then show all three products (A, B, C) in one plot with their forecasts. Let me:

    1. Query the most recent week of data for Products B and C
    2. Generate forecasts for Products B and C
    3. Visualize all three products together with their forecasts

    I already have the data for Product A. Let me query B and C first, then generate forecasts for both.

    Let me query both Products B and C simultaneously.
    </pre>
    </details>

    <p>Sure! Let me first fetch the data for Products B and C simultaneously.</p>

    <details>
    <summary>Ran select_query</summary>
    <p><strong>Assistant sent this info to clickhouse-playground</strong></p>
    <code>
    <pre class="tool">
    {
      "query": "\nSELECT timestamp, units_sold\nFROM default.sales\nWHERE product_id = 2\n  AND timestamp >= (SELECT max(timestamp) - INTERVAL 7 DAY FROM default.sales)\nORDER BY timestamp ASC\n"
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "columns": [
        "timestamp",
        "units_sold"
      ],
      "rows": [
        [
          "2026-03-08T15:00:00",
          44
        ],
        [
          "2026-03-08T16:00:00",
          41
        ],
        [
          "2026-03-08T17:00:00",
          47
        ],
        [
          "2026-03-08T18:00:00",
          37
        ],
        [
          "2026-03-08T19:00:00",
          39
        ],
        [
          "2026-03-08T20:00:00",
          32
        ],
        [
          "2026-03-08T21:00:00",
          22
        ],
        [
          "2026-03-08T22:00:00",
          20
        ],
        [
          "2026-03-08T23:00:00",
          17
        ],
        [
          "2026-03-09T00:00:00",
          11
        ],
        [
          "2026-03-09T01:00:00",
          14
        ],
        [
          "2026-03-09T02:00:00",
          7
        ],
        [
          "2026-03-09T03:00:00",
          3
        ],
        [
          "2026-03-09T04:00:00",
          6
        ],
        [
          "2026-03-09T05:00:00",
          7
        ],
        [
          "2026-03-09T06:00:00",
          11
        ],
        [
          "2026-03-09T07:00:00",
          17
        ],
        [
          "2026-03-09T08:00:00",
          18
        ],
        [
          "2026-03-09T09:00:00",
          21
        ],
        [
          "2026-03-09T10:00:00",
          30
        ],
        [
          "2026-03-09T11:00:00",
          35
        ],
        [
          "2026-03-09T12:00:00",
          38
        ],
        [
          "2026-03-09T13:00:00",
          41
        ],
        [
          "2026-03-09T14:00:00",
          46
        ],
        [
          "2026-03-09T15:00:00",
          47
        ],
        [
          "2026-03-09T16:00:00",
          43
        ],
        [
          "2026-03-09T17:00:00",
          41
        ],
        [
          "2026-03-09T18:00:00",
          39
        ],
        [
          "2026-03-09T19:00:00",
          33
        ],
        [
          "2026-03-09T20:00:00",
          33
        ],
        [
          "2026-03-09T21:00:00",
          26
        ],
        [
          "2026-03-09T22:00:00",
          20
        ],
        [
          "2026-03-09T23:00:00",
          17
        ],
        [
          "2026-03-10T00:00:00",
          7
        ],
        [
          "2026-03-10T01:00:00",
          9
        ],
        [
          "2026-03-10T02:00:00",
          11
        ],
        [
          "2026-03-10T03:00:00",
          7
        ],
        [
          "2026-03-10T04:00:00",
          4
        ],
        [
          "2026-03-10T05:00:00",
          9
        ],
        [
          "2026-03-10T06:00:00",
          11
        ],
        [
          "2026-03-10T07:00:00",
          15
        ],
        [
          "2026-03-10T08:00:00",
          18
        ],
        [
          "2026-03-10T09:00:00",
          24
        ],
        [
          "2026-03-10T10:00:00",
          29
        ],
        [
          "2026-03-10T11:00:00",
          39
        ],
        [
          "2026-03-10T12:00:00",
          42
        ],
        [
          "2026-03-10T13:00:00",
          42
        ],
        [
          "2026-03-10T14:00:00",
          46
        ],
        [
          "2026-03-10T15:00:00",
          42
        ],
        [
          "2026-03-10T16:00:00",
          45
        ],
        [
          "2026-03-10T17:00:00",
          46
        ],
        [
          "2026-03-10T18:00:00",
          39
        ],
        [
          "2026-03-10T19:00:00",
          37
        ],
        [
          "2026-03-10T20:00:00",
          32
        ],
        [
          "2026-03-10T21:00:00",
          27
        ],
        [
          "2026-03-10T22:00:00",
          21
        ],
        [
          "2026-03-10T23:00:00",
          15
        ],
        [
          "2026-03-11T00:00:00",
          12
        ],
        [
          "2026-03-11T01:00:00",
          9
        ],
        [
          "2026-03-11T02:00:00",
          6
        ],
        [
          "2026-03-11T03:00:00",
          7
        ],
        [
          "2026-03-11T04:00:00",
          6
        ],
        [
          "2026-03-11T05:00:00",
          9
        ],
        [
          "2026-03-11T06:00:00",
          12
        ],
        [
          "2026-03-11T07:00:00",
          16
        ],
        [
          "2026-03-11T08:00:00",
          22
        ],
        [
          "2026-03-11T09:00:00",
          26
        ],
        [
          "2026-03-11T10:00:00",
          30
        ],
        [
          "2026-03-11T11:00:00",
          33
        ],
        [
          "2026-03-11T12:00:00",
          34
        ],
        [
          "2026-03-11T13:00:00",
          40
        ],
        [
          "2026-03-11T14:00:00",
          42
        ],
        [
          "2026-03-11T15:00:00",
          44
        ],
        [
          "2026-03-11T16:00:00",
          48
        ],
        [
          "2026-03-11T17:00:00",
          42
        ],
        [
          "2026-03-11T18:00:00",
          39
        ],
        [
          "2026-03-11T19:00:00",
          36
        ],
        [
          "2026-03-11T20:00:00",
          28
        ],
        [
          "2026-03-11T21:00:00",
          27
        ],
        [
          "2026-03-11T22:00:00",
          19
        ],
        [
          "2026-03-11T23:00:00",
          16
        ],
        [
          "2026-03-12T00:00:00",
          8
        ],
        [
          "2026-03-12T01:00:00",
          10
        ],
        [
          "2026-03-12T02:00:00",
          5
        ],
        [
          "2026-03-12T03:00:00",
          3
        ],
        [
          "2026-03-12T04:00:00",
          7
        ],
        [
          "2026-03-12T05:00:00",
          9
        ],
        [
          "2026-03-12T06:00:00",
          9
        ],
        [
          "2026-03-12T07:00:00",
          13
        ],
        [
          "2026-03-12T08:00:00",
          17
        ],
        [
          "2026-03-12T09:00:00",
          24
        ],
        [
          "2026-03-12T10:00:00",
          31
        ],
        [
          "2026-03-12T11:00:00",
          38
        ],
        [
          "2026-03-12T12:00:00",
          36
        ],
        [
          "2026-03-12T13:00:00",
          42
        ],
        [
          "2026-03-12T14:00:00",
          44
        ],
        [
          "2026-03-12T15:00:00",
          46
        ],
        [
          "2026-03-12T16:00:00",
          43
        ],
        [
          "2026-03-12T17:00:00",
          48
        ],
        [
          "2026-03-12T18:00:00",
          39
        ],
        [
          "2026-03-12T19:00:00",
          35
        ],
        [
          "2026-03-12T20:00:00",
          31
        ],
        [
          "2026-03-12T21:00:00",
          24
        ],
        [
          "2026-03-12T22:00:00",
          16
        ],
        [
          "2026-03-12T23:00:00",
          15
        ],
        [
          "2026-03-13T00:00:00",
          15
        ],
        [
          "2026-03-13T01:00:00",
          11
        ],
        [
          "2026-03-13T02:00:00",
          8
        ],
        [
          "2026-03-13T03:00:00",
          7
        ],
        [
          "2026-03-13T04:00:00",
          6
        ],
        [
          "2026-03-13T05:00:00",
          9
        ],
        [
          "2026-03-13T06:00:00",
          9
        ],
        [
          "2026-03-13T07:00:00",
          16
        ],
        [
          "2026-03-13T08:00:00",
          17
        ],
        [
          "2026-03-13T09:00:00",
          27
        ],
        [
          "2026-03-13T10:00:00",
          30
        ],
        [
          "2026-03-13T11:00:00",
          34
        ],
        [
          "2026-03-13T12:00:00",
          40
        ],
        [
          "2026-03-13T13:00:00",
          44
        ],
        [
          "2026-03-13T14:00:00",
          42
        ],
        [
          "2026-03-13T15:00:00",
          48
        ],
        [
          "2026-03-13T16:00:00",
          44
        ],
        [
          "2026-03-13T17:00:00",
          43
        ],
        [
          "2026-03-13T18:00:00",
          39
        ],
        [
          "2026-03-13T19:00:00",
          32
        ],
        [
          "2026-03-13T20:00:00",
          30
        ],
        [
          "2026-03-13T21:00:00",
          23
        ],
        [
          "2026-03-13T22:00:00",
          20
        ],
        [
          "2026-03-13T23:00:00",
          15
        ],
        [
          "2026-03-14T00:00:00",
          9
        ],
        [
          "2026-03-14T01:00:00",
          8
        ],
        [
          "2026-03-14T02:00:00",
          5
        ],
        [
          "2026-03-14T03:00:00",
          7
        ],
        [
          "2026-03-14T04:00:00",
          8
        ],
        [
          "2026-03-14T05:00:00",
          7
        ],
        [
          "2026-03-14T06:00:00",
          13
        ],
        [
          "2026-03-14T07:00:00",
          19
        ],
        [
          "2026-03-14T08:00:00",
          19
        ],
        [
          "2026-03-14T09:00:00",
          25
        ],
        [
          "2026-03-14T10:00:00",
          31
        ],
        [
          "2026-03-14T11:00:00",
          35
        ],
        [
          "2026-03-14T12:00:00",
          40
        ],
        [
          "2026-03-14T13:00:00",
          43
        ],
        [
          "2026-03-14T14:00:00",
          45
        ],
        [
          "2026-03-14T15:00:00",
          44
        ],
        [
          "2026-03-14T16:00:00",
          44
        ],
        [
          "2026-03-14T17:00:00",
          44
        ],
        [
          "2026-03-14T18:00:00",
          42
        ],
        [
          "2026-03-14T19:00:00",
          32
        ],
        [
          "2026-03-14T20:00:00",
          31
        ],
        [
          "2026-03-14T21:00:00",
          26
        ],
        [
          "2026-03-14T22:00:00",
          20
        ],
        [
          "2026-03-14T23:00:00",
          13
        ],
        [
          "2026-03-15T00:00:00",
          10
        ],
        [
          "2026-03-15T01:00:00",
          10
        ],
        [
          "2026-03-15T02:00:00",
          7
        ],
        [
          "2026-03-15T03:00:00",
          8
        ],
        [
          "2026-03-15T04:00:00",
          12
        ],
        [
          "2026-03-15T05:00:00",
          10
        ],
        [
          "2026-03-15T06:00:00",
          13
        ],
        [
          "2026-03-15T07:00:00",
          14
        ],
        [
          "2026-03-15T08:00:00",
          18
        ],
        [
          "2026-03-15T09:00:00",
          24
        ],
        [
          "2026-03-15T10:00:00",
          32
        ],
        [
          "2026-03-15T11:00:00",
          34
        ],
        [
          "2026-03-15T12:00:00",
          39
        ],
        [
          "2026-03-15T13:00:00",
          39
        ],
        [
          "2026-03-15T14:00:00",
          41
        ],
        [
          "2026-03-15T15:00:00",
          47
        ]
      ]
    }
    </pre>
    </code>
    </details>

    <details>
    <summary>Ran select_query</summary>
    <p><strong>Assistant sent this info to clickhouse-playground</strong></p>
    <code>
    <pre class="tool">
    {
      "query": "\nSELECT timestamp, units_sold\nFROM default.sales\nWHERE product_id = 3\n  AND timestamp >= (SELECT max(timestamp) - INTERVAL 7 DAY FROM default.sales)\nORDER BY timestamp ASC\n"
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "columns": [
        "timestamp",
        "units_sold"
      ],
      "rows": [
        [
          "2026-03-08T15:00:00",
          67
        ],
        [
          "2026-03-08T16:00:00",
          65
        ],
        [
          "2026-03-08T17:00:00",
          66
        ],
        [
          "2026-03-08T18:00:00",
          64
        ],
        [
          "2026-03-08T19:00:00",
          51
        ],
        [
          "2026-03-08T20:00:00",
          51
        ],
        [
          "2026-03-08T21:00:00",
          42
        ],
        [
          "2026-03-08T22:00:00",
          30
        ],
        [
          "2026-03-08T23:00:00",
          18
        ],
        [
          "2026-03-09T00:00:00",
          11
        ],
        [
          "2026-03-09T01:00:00",
          16
        ],
        [
          "2026-03-09T02:00:00",
          9
        ],
        [
          "2026-03-09T03:00:00",
          1
        ],
        [
          "2026-03-09T04:00:00",
          13
        ],
        [
          "2026-03-09T05:00:00",
          4
        ],
        [
          "2026-03-09T06:00:00",
          19
        ],
        [
          "2026-03-09T07:00:00",
          24
        ],
        [
          "2026-03-09T08:00:00",
          33
        ],
        [
          "2026-03-09T09:00:00",
          35
        ],
        [
          "2026-03-09T10:00:00",
          39
        ],
        [
          "2026-03-09T11:00:00",
          52
        ],
        [
          "2026-03-09T12:00:00",
          59
        ],
        [
          "2026-03-09T13:00:00",
          65
        ],
        [
          "2026-03-09T14:00:00",
          70
        ],
        [
          "2026-03-09T15:00:00",
          67
        ],
        [
          "2026-03-09T16:00:00",
          67
        ],
        [
          "2026-03-09T17:00:00",
          63
        ],
        [
          "2026-03-09T18:00:00",
          58
        ],
        [
          "2026-03-09T19:00:00",
          53
        ],
        [
          "2026-03-09T20:00:00",
          45
        ],
        [
          "2026-03-09T21:00:00",
          37
        ],
        [
          "2026-03-09T22:00:00",
          28
        ],
        [
          "2026-03-09T23:00:00",
          24
        ],
        [
          "2026-03-10T00:00:00",
          18
        ],
        [
          "2026-03-10T01:00:00",
          8
        ],
        [
          "2026-03-10T02:00:00",
          11
        ],
        [
          "2026-03-10T03:00:00",
          2
        ],
        [
          "2026-03-10T04:00:00",
          14
        ],
        [
          "2026-03-10T05:00:00",
          13
        ],
        [
          "2026-03-10T06:00:00",
          13
        ],
        [
          "2026-03-10T07:00:00",
          23
        ],
        [
          "2026-03-10T08:00:00",
          30
        ],
        [
          "2026-03-10T09:00:00",
          39
        ],
        [
          "2026-03-10T10:00:00",
          44
        ],
        [
          "2026-03-10T11:00:00",
          50
        ],
        [
          "2026-03-10T12:00:00",
          58
        ],
        [
          "2026-03-10T13:00:00",
          62
        ],
        [
          "2026-03-10T14:00:00",
          65
        ],
        [
          "2026-03-10T15:00:00",
          73
        ],
        [
          "2026-03-10T16:00:00",
          64
        ],
        [
          "2026-03-10T17:00:00",
          63
        ],
        [
          "2026-03-10T18:00:00",
          67
        ],
        [
          "2026-03-10T19:00:00",
          53
        ],
        [
          "2026-03-10T20:00:00",
          44
        ],
        [
          "2026-03-10T21:00:00",
          38
        ],
        [
          "2026-03-10T22:00:00",
          32
        ],
        [
          "2026-03-10T23:00:00",
          20
        ],
        [
          "2026-03-11T00:00:00",
          17
        ],
        [
          "2026-03-11T01:00:00",
          12
        ],
        [
          "2026-03-11T02:00:00",
          9
        ],
        [
          "2026-03-11T03:00:00",
          3
        ],
        [
          "2026-03-11T04:00:00",
          16
        ],
        [
          "2026-03-11T05:00:00",
          13
        ],
        [
          "2026-03-11T06:00:00",
          16
        ],
        [
          "2026-03-11T07:00:00",
          29
        ],
        [
          "2026-03-11T08:00:00",
          28
        ],
        [
          "2026-03-11T09:00:00",
          38
        ],
        [
          "2026-03-11T10:00:00",
          47
        ],
        [
          "2026-03-11T11:00:00",
          46
        ],
        [
          "2026-03-11T12:00:00",
          59
        ],
        [
          "2026-03-11T13:00:00",
          63
        ],
        [
          "2026-03-11T14:00:00",
          66
        ],
        [
          "2026-03-11T15:00:00",
          68
        ],
        [
          "2026-03-11T16:00:00",
          69
        ],
        [
          "2026-03-11T17:00:00",
          62
        ],
        [
          "2026-03-11T18:00:00",
          55
        ],
        [
          "2026-03-11T19:00:00",
          47
        ],
        [
          "2026-03-11T20:00:00",
          47
        ],
        [
          "2026-03-11T21:00:00",
          37
        ],
        [
          "2026-03-11T22:00:00",
          28
        ],
        [
          "2026-03-11T23:00:00",
          23
        ],
        [
          "2026-03-12T00:00:00",
          20
        ],
        [
          "2026-03-12T01:00:00",
          12
        ],
        [
          "2026-03-12T02:00:00",
          8
        ],
        [
          "2026-03-12T03:00:00",
          6
        ],
        [
          "2026-03-12T04:00:00",
          7
        ],
        [
          "2026-03-12T05:00:00",
          13
        ],
        [
          "2026-03-12T06:00:00",
          18
        ],
        [
          "2026-03-12T07:00:00",
          18
        ],
        [
          "2026-03-12T08:00:00",
          26
        ],
        [
          "2026-03-12T09:00:00",
          40
        ],
        [
          "2026-03-12T10:00:00",
          44
        ],
        [
          "2026-03-12T11:00:00",
          52
        ],
        [
          "2026-03-12T12:00:00",
          60
        ],
        [
          "2026-03-12T13:00:00",
          64
        ],
        [
          "2026-03-12T14:00:00",
          65
        ],
        [
          "2026-03-12T15:00:00",
          67
        ],
        [
          "2026-03-12T16:00:00",
          64
        ],
        [
          "2026-03-12T17:00:00",
          68
        ],
        [
          "2026-03-12T18:00:00",
          57
        ],
        [
          "2026-03-12T19:00:00",
          55
        ],
        [
          "2026-03-12T20:00:00",
          50
        ],
        [
          "2026-03-12T21:00:00",
          36
        ],
        [
          "2026-03-12T22:00:00",
          28
        ],
        [
          "2026-03-12T23:00:00",
          21
        ],
        [
          "2026-03-13T00:00:00",
          16
        ],
        [
          "2026-03-13T01:00:00",
          16
        ],
        [
          "2026-03-13T02:00:00",
          13
        ],
        [
          "2026-03-13T03:00:00",
          6
        ],
        [
          "2026-03-13T04:00:00",
          9
        ],
        [
          "2026-03-13T05:00:00",
          14
        ],
        [
          "2026-03-13T06:00:00",
          18
        ],
        [
          "2026-03-13T07:00:00",
          22
        ],
        [
          "2026-03-13T08:00:00",
          26
        ],
        [
          "2026-03-13T09:00:00",
          36
        ],
        [
          "2026-03-13T10:00:00",
          48
        ],
        [
          "2026-03-13T11:00:00",
          53
        ],
        [
          "2026-03-13T12:00:00",
          59
        ],
        [
          "2026-03-13T13:00:00",
          64
        ],
        [
          "2026-03-13T14:00:00",
          70
        ],
        [
          "2026-03-13T15:00:00",
          70
        ],
        [
          "2026-03-13T16:00:00",
          72
        ],
        [
          "2026-03-13T17:00:00",
          62
        ],
        [
          "2026-03-13T18:00:00",
          61
        ],
        [
          "2026-03-13T19:00:00",
          52
        ],
        [
          "2026-03-13T20:00:00",
          41
        ],
        [
          "2026-03-13T21:00:00",
          35
        ],
        [
          "2026-03-13T22:00:00",
          29
        ],
        [
          "2026-03-13T23:00:00",
          22
        ],
        [
          "2026-03-14T00:00:00",
          19
        ],
        [
          "2026-03-14T01:00:00",
          7
        ],
        [
          "2026-03-14T02:00:00",
          9
        ],
        [
          "2026-03-14T03:00:00",
          8
        ],
        [
          "2026-03-14T04:00:00",
          5
        ],
        [
          "2026-03-14T05:00:00",
          19
        ],
        [
          "2026-03-14T06:00:00",
          16
        ],
        [
          "2026-03-14T07:00:00",
          20
        ],
        [
          "2026-03-14T08:00:00",
          25
        ],
        [
          "2026-03-14T09:00:00",
          36
        ],
        [
          "2026-03-14T10:00:00",
          48
        ],
        [
          "2026-03-14T11:00:00",
          55
        ],
        [
          "2026-03-14T12:00:00",
          57
        ],
        [
          "2026-03-14T13:00:00",
          58
        ],
        [
          "2026-03-14T14:00:00",
          68
        ],
        [
          "2026-03-14T15:00:00",
          70
        ],
        [
          "2026-03-14T16:00:00",
          65
        ],
        [
          "2026-03-14T17:00:00",
          61
        ],
        [
          "2026-03-14T18:00:00",
          55
        ],
        [
          "2026-03-14T19:00:00",
          52
        ],
        [
          "2026-03-14T20:00:00",
          41
        ],
        [
          "2026-03-14T21:00:00",
          34
        ],
        [
          "2026-03-14T22:00:00",
          30
        ],
        [
          "2026-03-14T23:00:00",
          24
        ],
        [
          "2026-03-15T00:00:00",
          21
        ],
        [
          "2026-03-15T01:00:00",
          15
        ],
        [
          "2026-03-15T02:00:00",
          5
        ],
        [
          "2026-03-15T03:00:00",
          9
        ],
        [
          "2026-03-15T04:00:00",
          9
        ],
        [
          "2026-03-15T05:00:00",
          14
        ],
        [
          "2026-03-15T06:00:00",
          19
        ],
        [
          "2026-03-15T07:00:00",
          28
        ],
        [
          "2026-03-15T08:00:00",
          31
        ],
        [
          "2026-03-15T09:00:00",
          41
        ],
        [
          "2026-03-15T10:00:00",
          50
        ],
        [
          "2026-03-15T11:00:00",
          51
        ],
        [
          "2026-03-15T12:00:00",
          64
        ],
        [
          "2026-03-15T13:00:00",
          67
        ],
        [
          "2026-03-15T14:00:00",
          69
        ],
        [
          "2026-03-15T15:00:00",
          68
        ]
      ]
    }
    </pre>
    </code>
    </details>

    <p>Got the data for both products! Now let me generate forecasts for Products B and C simultaneously.</p>

    <details>
    <summary>Ran generate_forecasts</summary>
    <p><strong>Assistant sent this info to chronos-forecasting</strong></p>
    <code>
    <pre class="tool">
    {
      "target": [
        44,
        41,
        47,
        37,
        39,
        32,
        22,
        20,
        17,
        11,
        14,
        7,
        3,
        6,
        7,
        11,
        17,
        18,
        21,
        30,
        35,
        38,
        41,
        46,
        47,
        43,
        41,
        39,
        33,
        33,
        26,
        20,
        17,
        7,
        9,
        11,
        7,
        4,
        9,
        11,
        15,
        18,
        24,
        29,
        39,
        42,
        42,
        46,
        42,
        45,
        46,
        39,
        37,
        32,
        27,
        21,
        15,
        12,
        9,
        6,
        7,
        6,
        9,
        12,
        16,
        22,
        26,
        30,
        33,
        34,
        40,
        42,
        44,
        48,
        42,
        39,
        36,
        28,
        27,
        19,
        16,
        8,
        10,
        5,
        3,
        7,
        9,
        9,
        13,
        17,
        24,
        31,
        38,
        36,
        42,
        44,
        46,
        43,
        48,
        39,
        35,
        31,
        24,
        16,
        15,
        15,
        11,
        8,
        7,
        6,
        9,
        9,
        16,
        17,
        27,
        30,
        34,
        40,
        44,
        42,
        48,
        44,
        43,
        39,
        32,
        30,
        23,
        20,
        15,
        9,
        8,
        5,
        7,
        8,
        7,
        13,
        19,
        19,
        25,
        31,
        35,
        40,
        43,
        45,
        44,
        44,
        44,
        42,
        32,
        31,
        26,
        20,
        13,
        10,
        10,
        7,
        8,
        12,
        10,
        13,
        14,
        18,
        24,
        32,
        34,
        39,
        39,
        41,
        47
      ],
      "prediction_length": 48,
      "quantile_levels": [
        0.1,
        0.5,
        0.9
      ]
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "mean": [
        45.744361877441406,
        44.47789764404297,
        39.96786880493164,
        34.93931579589844,
        30.40752410888672,
        25.078933715820312,
        19.764469146728516,
        15.15910816192627,
        11.105031967163086,
        9.184202194213867,
        7.405527114868164,
        6.835023880004883,
        7.098142623901367,
        8.534454345703125,
        11.933398246765137,
        16.158313751220703,
        19.992860794067383,
        25.135028839111328,
        30.816650390625,
        35.86872863769531,
        39.165767669677734,
        42.46210479736328,
        44.626007080078125,
        45.996063232421875,
        45.7534065246582,
        44.59668731689453,
        40.11052703857422,
        35.24456787109375,
        31.033628463745117,
        25.713680267333984,
        20.289962768554688,
        15.610541343688965,
        11.549332618713379,
        9.544018745422363,
        7.7064056396484375,
        6.958499908447266,
        7.410867691040039,
        9.032258987426758,
        12.31091022491455,
        16.357358932495117,
        20.261585235595703,
        25.50418472290039,
        31.366159439086914,
        36.46283721923828,
        39.60572052001953,
        42.69723129272461,
        44.79161834716797,
        46.19174575805664
      ],
      "0.1": [
        43.202728271484375,
        41.657466888427734,
        37.34388732910156,
        32.49301528930664,
        27.895044326782227,
        22.69713592529297,
        17.48502540588379,
        12.914217948913574,
        8.867912292480469,
        6.805437088012695,
        5.133642196655273,
        4.6055755615234375,
        4.980224609375,
        6.566743850708008,
        9.907864570617676,
        14.042644500732422,
        17.96416664123535,
        23.004030227661133,
        28.466156005859375,
        33.398677825927734,
        36.72842788696289,
        39.793521881103516,
        41.68659973144531,
        42.83111572265625,
        42.48583984375,
        41.06452941894531,
        36.83848571777344,
        32.41840744018555,
        28.241992950439453,
        23.26666259765625,
        18.005842208862305,
        13.322606086730957,
        9.245460510253906,
        7.142238616943359,
        5.440708160400391,
        4.845056533813477,
        5.325952529907227,
        7.058143615722656,
        10.134407043457031,
        14.08029842376709,
        18.03704071044922,
        23.15070152282715,
        28.727794647216797,
        33.606204986572266,
        36.765869140625,
        39.615299224853516,
        41.445343017578125,
        42.517974853515625
      ],
      "0.5": [
        45.744361877441406,
        44.47789764404297,
        39.96786880493164,
        34.93931579589844,
        30.40752410888672,
        25.078933715820312,
        19.764469146728516,
        15.15910816192627,
        11.105031967163086,
        9.184202194213867,
        7.405527114868164,
        6.835023880004883,
        7.098142623901367,
        8.534454345703125,
        11.933398246765137,
        16.158313751220703,
        19.992860794067383,
        25.135028839111328,
        30.816650390625,
        35.86872863769531,
        39.165767669677734,
        42.46210479736328,
        44.626007080078125,
        45.996063232421875,
        45.7534065246582,
        44.59668731689453,
        40.11052703857422,
        35.24456787109375,
        31.033628463745117,
        25.713680267333984,
        20.289962768554688,
        15.610541343688965,
        11.549332618713379,
        9.544018745422363,
        7.7064056396484375,
        6.958499908447266,
        7.410867691040039,
        9.032258987426758,
        12.31091022491455,
        16.357358932495117,
        20.261585235595703,
        25.50418472290039,
        31.366159439086914,
        36.46283721923828,
        39.60572052001953,
        42.69723129272461,
        44.79161834716797,
        46.19174575805664
      ],
      "0.9": [
        48.60978698730469,
        47.43096923828125,
        42.79112243652344,
        37.6799201965332,
        33.3004150390625,
        27.90454864501953,
        22.537134170532227,
        17.90163230895996,
        13.724783897399902,
        11.975788116455078,
        9.977222442626953,
        9.093873977661133,
        9.463135719299316,
        10.984129905700684,
        14.422269821166992,
        18.565746307373047,
        22.39069175720215,
        27.66008949279785,
        33.55706024169922,
        38.70289611816406,
        42.075965881347656,
        45.412227630615234,
        47.52215576171875,
        48.96488952636719,
        48.8265380859375,
        47.848453521728516,
        43.23004913330078,
        38.305755615234375,
        34.17613983154297,
        28.809988021850586,
        23.35015106201172,
        18.584577560424805,
        14.49343204498291,
        12.689022064208984,
        10.720396995544434,
        9.662089347839355,
        10.225909233093262,
        11.88810920715332,
        15.130329132080078,
        19.02608871459961,
        22.825077056884766,
        28.250431060791016,
        34.244171142578125,
        39.468650817871094,
        42.69873046875,
        45.907142639160156,
        47.94013214111328,
        49.381919860839844
      ]
    }
    </pre>
    </code>
    </details>

    <details>
    <summary>Ran generate_forecasts</summary>
    <p><strong>Assistant sent this info to chronos-forecasting</strong></p>
    <code>
    <pre class="tool">
    {
      "target": [
        67,
        65,
        66,
        64,
        51,
        51,
        42,
        30,
        18,
        11,
        16,
        9,
        1,
        13,
        4,
        19,
        24,
        33,
        35,
        39,
        52,
        59,
        65,
        70,
        67,
        67,
        63,
        58,
        53,
        45,
        37,
        28,
        24,
        18,
        8,
        11,
        2,
        14,
        13,
        13,
        23,
        30,
        39,
        44,
        50,
        58,
        62,
        65,
        73,
        64,
        63,
        67,
        53,
        44,
        38,
        32,
        20,
        17,
        12,
        9,
        3,
        16,
        13,
        16,
        29,
        28,
        38,
        47,
        46,
        59,
        63,
        66,
        68,
        69,
        62,
        55,
        47,
        47,
        37,
        28,
        23,
        20,
        12,
        8,
        6,
        7,
        13,
        18,
        18,
        26,
        40,
        44,
        52,
        60,
        64,
        65,
        67,
        64,
        68,
        57,
        55,
        50,
        36,
        28,
        21,
        16,
        16,
        13,
        6,
        9,
        14,
        18,
        22,
        26,
        36,
        48,
        53,
        59,
        64,
        70,
        70,
        72,
        62,
        61,
        52,
        41,
        35,
        29,
        22,
        19,
        7,
        9,
        8,
        5,
        19,
        16,
        20,
        25,
        36,
        48,
        55,
        57,
        58,
        68,
        70,
        65,
        61,
        55,
        52,
        41,
        34,
        30,
        24,
        21,
        15,
        5,
        9,
        9,
        14,
        19,
        28,
        31,
        41,
        50,
        51,
        64,
        67,
        69,
        68
      ],
      "prediction_length": 48,
      "quantile_levels": [
        0.1,
        0.5,
        0.9
      ]
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    {
      "mean": [
        68.25753784179688,
        64.33294677734375,
        58.953407287597656,
        52.182167053222656,
        44.8023567199707,
        36.48902130126953,
        29.293916702270508,
        23.117162704467773,
        18.24557876586914,
        12.591239929199219,
        9.324899673461914,
        7.880804061889648,
        10.169549942016602,
        14.116628646850586,
        18.203784942626953,
        24.026535034179688,
        30.23114776611328,
        38.921058654785156,
        47.02241897583008,
        53.7126350402832,
        60.47807312011719,
        64.84934997558594,
        68.66142272949219,
        70.09188842773438,
        68.77432250976562,
        64.90254974365234,
        59.55329132080078,
        52.759666442871094,
        45.605072021484375,
        37.1990966796875,
        29.969520568847656,
        23.500141143798828,
        18.895471572875977,
        13.273134231567383,
        9.766721725463867,
        7.782873153686523,
        10.10135269165039,
        14.322179794311523,
        18.6790828704834,
        24.71501922607422,
        31.01270866394043,
        39.43730163574219,
        47.36338806152344,
        54.099761962890625,
        61.12646484375,
        65.52592468261719,
        69.14380645751953,
        70.39305114746094
      ],
      "0.1": [
        65.0497055053711,
        61.02140808105469,
        55.667762756347656,
        49.130184173583984,
        41.937583923339844,
        33.930274963378906,
        26.850168228149414,
        20.65131950378418,
        15.420270919799805,
        9.610015869140625,
        6.014139175415039,
        4.563690185546875,
        6.7572479248046875,
        10.61678695678711,
        15.046772003173828,
        20.946298599243164,
        27.35129737854004,
        35.78847122192383,
        43.83382034301758,
        50.54799270629883,
        56.998382568359375,
        61.15869140625,
        64.4871597290039,
        65.62289428710938,
        64.30809020996094,
        60.597747802734375,
        55.28544616699219,
        49.05252456665039,
        42.306427001953125,
        34.51346206665039,
        27.43219757080078,
        21.02750015258789,
        15.994155883789062,
        10.284709930419922,
        6.550397872924805,
        4.792110443115234,
        6.905149459838867,
        10.953042984008789,
        15.455942153930664,
        21.445810317993164,
        27.832324981689453,
        35.966773986816406,
        43.770023345947266,
        50.32490539550781,
        56.82306671142578,
        60.98479080200195,
        64.19062805175781,
        65.11436462402344
      ],
      "0.5": [
        68.25753784179688,
        64.33294677734375,
        58.953407287597656,
        52.182167053222656,
        44.8023567199707,
        36.48902130126953,
        29.293916702270508,
        23.117162704467773,
        18.24557876586914,
        12.591239929199219,
        9.324899673461914,
        7.880804061889648,
        10.169549942016602,
        14.116628646850586,
        18.203784942626953,
        24.026535034179688,
        30.23114776611328,
        38.921058654785156,
        47.02241897583008,
        53.7126350402832,
        60.47807312011719,
        64.84934997558594,
        68.66142272949219,
        70.09188842773438,
        68.77432250976562,
        64.90254974365234,
        59.55329132080078,
        52.759666442871094,
        45.605072021484375,
        37.1990966796875,
        29.969520568847656,
        23.500141143798828,
        18.895471572875977,
        13.273134231567383,
        9.766721725463867,
        7.782873153686523,
        10.10135269165039,
        14.322179794311523,
        18.6790828704834,
        24.71501922607422,
        31.01270866394043,
        39.43730163574219,
        47.36338806152344,
        54.099761962890625,
        61.12646484375,
        65.52592468261719,
        69.14380645751953,
        70.39305114746094
      ],
      "0.9": [
        71.67506408691406,
        67.64664459228516,
        62.195289611816406,
        55.31642150878906,
        47.960060119628906,
        39.451725006103516,
        32.17584228515625,
        25.903423309326172,
        21.21776580810547,
        15.803155899047852,
        12.547943115234375,
        10.65243148803711,
        13.38730239868164,
        17.783737182617188,
        21.71316146850586,
        27.42102813720703,
        33.64549255371094,
        42.66716766357422,
        50.950584411621094,
        57.488285064697266,
        64.4905014038086,
        68.61067199707031,
        72.39262390136719,
        73.94755554199219,
        72.77479553222656,
        68.87016296386719,
        63.50397872924805,
        56.57518768310547,
        49.40583038330078,
        40.81800842285156,
        33.40961456298828,
        26.681062698364258,
        22.37085723876953,
        17.016590118408203,
        13.691097259521484,
        11.294649124145508,
        13.996709823608398,
        18.501005172729492,
        22.67669677734375,
        28.56482696533203,
        34.7428092956543,
        43.582984924316406,
        51.486045837402344,
        58.23878479003906,
        65.67634582519531,
        69.92729187011719,
        73.41410827636719,
        74.68563079833984
      ]
    }
    </pre>
    </code>
    </details>

    <br>

    <details>
    <summary>Thoughts</summary>
    <pre class="thought">
    Now I have all the data and forecasts for Products B and C. I also have the data and forecasts for Product A from the previous context. Let me now visualize all three products together with their forecasts.

    For Product A, I need to use the data and forecast from the previous context:
    - Data rows from previous context
    - Forecast from previous context

    Let me construct the timestamps for the forecasts. The last timestamp in the data is "2026-03-15T15:00:00", so the forecast starts at "2026-03-15T16:00:00" and goes for 48 hours hourly.

    Let me generate the forecast timestamps:

    I'm creating the 48-hour forecast timeline starting from 2026-03-15T16:00:00, incrementing by one hour for each step through 2026-03-17T15:00:00. Continuing through the hourly timestamps into mid-March 16th. Continuing through the hourly timestamps into the next day, tracking the progression from early morning through mid-afternoon on March 17th.

    Now I'm gathering the historical data and forecast values for all three products to pass into the visualization function—I have the actual readings for Product A along with its predicted mean and quantile values across the forecast period. Looking at performance metrics across different product variants and their temperature curves. Now I'm looking at another set of numerical values, likely representing similar measurements or coordinates across a sequence of points. Compiling performance metrics for Product A... I'm continuing to compile the hourly data points for Product A, tracking the values through the night and into the next morning. Continuing through the hourly data for March 10th, tracking the temperature readings as they dip to their lowest point around 4 AM before climbing back up through the morning hours. Generating the time series data... Continuing through the hourly data... Generating time series data... Continuing through the hourly data for March 14th... Now I'm structuring the data object to include all three data series along with their corresponding forecast values for the chart visualization.
    </pre>
    </details>

    <p>
    Excellent! Now let me put everything together and visualize all three products with their forecasts in one combined plot.
    </p>

    <details>
    <summary>Ran visualize_data</summary>
    <p><strong>Assistant sent this info to data-visualization</strong></p>
    <code>
    <pre class="tool">
    {
      "inputs": {
        "data": {
          "Product A": {
            "columns": [
              "timestamp",
              "units_sold"
            ],
            "rows": [
              [
                "2026-03-08T15:00:00",
                21
              ],
              [
                "2026-03-08T16:00:00",
                20
              ],
              [
                "2026-03-08T17:00:00",
                21
              ],
              [
                "2026-03-08T18:00:00",
                25
              ],
              [
                "2026-03-08T19:00:00",
                23
              ],
              [
                "2026-03-08T20:00:00",
                24
              ],
              [
                "2026-03-08T21:00:00",
                26
              ],
              [
                "2026-03-08T22:00:00",
                25
              ],
              [
                "2026-03-08T23:00:00",
                17
              ],
              [
                "2026-03-09T00:00:00",
                13
              ],
              [
                "2026-03-09T01:00:00",
                7
              ],
              [
                "2026-03-09T02:00:00",
                4
              ],
              [
                "2026-03-09T03:00:00",
                2
              ],
              [
                "2026-03-09T04:00:00",
                7
              ],
              [
                "2026-03-09T05:00:00",
                14
              ],
              [
                "2026-03-09T06:00:00",
                22
              ],
              [
                "2026-03-09T07:00:00",
                26
              ],
              [
                "2026-03-09T08:00:00",
                35
              ],
              [
                "2026-03-09T09:00:00",
                37
              ],
              [
                "2026-03-09T10:00:00",
                39
              ],
              [
                "2026-03-09T11:00:00",
                36
              ],
              [
                "2026-03-09T12:00:00",
                32
              ],
              [
                "2026-03-09T13:00:00",
                27
              ],
              [
                "2026-03-09T14:00:00",
                23
              ],
              [
                "2026-03-09T15:00:00",
                20
              ],
              [
                "2026-03-09T16:00:00",
                19
              ],
              [
                "2026-03-09T17:00:00",
                20
              ],
              [
                "2026-03-09T18:00:00",
                22
              ],
              [
                "2026-03-09T19:00:00",
                25
              ],
              [
                "2026-03-09T20:00:00",
                26
              ],
              [
                "2026-03-09T21:00:00",
                27
              ],
              [
                "2026-03-09T22:00:00",
                23
              ],
              [
                "2026-03-09T23:00:00",
                19
              ],
              [
                "2026-03-10T00:00:00",
                12
              ],
              [
                "2026-03-10T01:00:00",
                8
              ],
              [
                "2026-03-10T02:00:00",
                4
              ],
              [
                "2026-03-10T03:00:00",
                4
              ],
              [
                "2026-03-10T04:00:00",
                9
              ],
              [
                "2026-03-10T05:00:00",
                12
              ],
              [
                "2026-03-10T06:00:00",
                20
              ],
              [
                "2026-03-10T07:00:00",
                28
              ],
              [
                "2026-03-10T08:00:00",
                35
              ],
              [
                "2026-03-10T09:00:00",
                36
              ],
              [
                "2026-03-10T10:00:00",
                39
              ],
              [
                "2026-03-10T11:00:00",
                36
              ],
              [
                "2026-03-10T12:00:00",
                34
              ],
              [
                "2026-03-10T13:00:00",
                26
              ],
              [
                "2026-03-10T14:00:00",
                22
              ],
              [
                "2026-03-10T15:00:00",
                19
              ],
              [
                "2026-03-10T16:00:00",
                21
              ],
              [
                "2026-03-10T17:00:00",
                22
              ],
              [
                "2026-03-10T18:00:00",
                23
              ],
              [
                "2026-03-10T19:00:00",
                25
              ],
              [
                "2026-03-10T20:00:00",
                28
              ],
              [
                "2026-03-10T21:00:00",
                27
              ],
              [
                "2026-03-10T22:00:00",
                23
              ],
              [
                "2026-03-10T23:00:00",
                17
              ],
              [
                "2026-03-11T00:00:00",
                11
              ],
              [
                "2026-03-11T01:00:00",
                9
              ],
              [
                "2026-03-11T02:00:00",
                4
              ],
              [
                "2026-03-11T03:00:00",
                3
              ],
              [
                "2026-03-11T04:00:00",
                6
              ],
              [
                "2026-03-11T05:00:00",
                11
              ],
              [
                "2026-03-11T06:00:00",
                22
              ],
              [
                "2026-03-11T07:00:00",
                29
              ],
              [
                "2026-03-11T08:00:00",
                33
              ],
              [
                "2026-03-11T09:00:00",
                38
              ],
              [
                "2026-03-11T10:00:00",
                38
              ],
              [
                "2026-03-11T11:00:00",
                37
              ],
              [
                "2026-03-11T12:00:00",
                32
              ],
              [
                "2026-03-11T13:00:00",
                26
              ],
              [
                "2026-03-11T14:00:00",
                24
              ],
              [
                "2026-03-11T15:00:00",
                21
              ],
              [
                "2026-03-11T16:00:00",
                21
              ],
              [
                "2026-03-11T17:00:00",
                20
              ],
              [
                "2026-03-11T18:00:00",
                24
              ],
              [
                "2026-03-11T19:00:00",
                27
              ],
              [
                "2026-03-11T20:00:00",
                28
              ],
              [
                "2026-03-11T21:00:00",
                27
              ],
              [
                "2026-03-11T22:00:00",
                23
              ],
              [
                "2026-03-11T23:00:00",
                18
              ],
              [
                "2026-03-12T00:00:00",
                11
              ],
              [
                "2026-03-12T01:00:00",
                8
              ],
              [
                "2026-03-12T02:00:00",
                3
              ],
              [
                "2026-03-12T03:00:00",
                4
              ],
              [
                "2026-03-12T04:00:00",
                7
              ],
              [
                "2026-03-12T05:00:00",
                13
              ],
              [
                "2026-03-12T06:00:00",
                19
              ],
              [
                "2026-03-12T07:00:00",
                29
              ],
              [
                "2026-03-12T08:00:00",
                34
              ],
              [
                "2026-03-12T09:00:00",
                41
              ],
              [
                "2026-03-12T10:00:00",
                38
              ],
              [
                "2026-03-12T11:00:00",
                37
              ],
              [
                "2026-03-12T12:00:00",
                31
              ],
              [
                "2026-03-12T13:00:00",
                29
              ],
              [
                "2026-03-12T14:00:00",
                26
              ],
              [
                "2026-03-12T15:00:00",
                21
              ],
              [
                "2026-03-12T16:00:00",
                19
              ],
              [
                "2026-03-12T17:00:00",
                21
              ],
              [
                "2026-03-12T18:00:00",
                22
              ],
              [
                "2026-03-12T19:00:00",
                25
              ],
              [
                "2026-03-12T20:00:00",
                26
              ],
              [
                "2026-03-12T21:00:00",
                25
              ],
              [
                "2026-03-12T22:00:00",
                21
              ],
              [
                "2026-03-12T23:00:00",
                17
              ],
              [
                "2026-03-13T00:00:00",
                15
              ],
              [
                "2026-03-13T01:00:00",
                6
              ],
              [
                "2026-03-13T02:00:00",
                4
              ],
              [
                "2026-03-13T03:00:00",
                6
              ],
              [
                "2026-03-13T04:00:00",
                7
              ],
              [
                "2026-03-13T05:00:00",
                13
              ],
              [
                "2026-03-13T06:00:00",
                21
              ],
              [
                "2026-03-13T07:00:00",
                27
              ],
              [
                "2026-03-13T08:00:00",
                33
              ],
              [
                "2026-03-13T09:00:00",
                38
              ],
              [
                "2026-03-13T10:00:00",
                38
              ],
              [
                "2026-03-13T11:00:00",
                36
              ],
              [
                "2026-03-13T12:00:00",
                33
              ],
              [
                "2026-03-13T13:00:00",
                25
              ],
              [
                "2026-03-13T14:00:00",
                23
              ],
              [
                "2026-03-13T15:00:00",
                21
              ],
              [
                "2026-03-13T16:00:00",
                20
              ],
              [
                "2026-03-13T17:00:00",
                23
              ],
              [
                "2026-03-13T18:00:00",
                23
              ],
              [
                "2026-03-13T19:00:00",
                26
              ],
              [
                "2026-03-13T20:00:00",
                28
              ],
              [
                "2026-03-13T21:00:00",
                25
              ],
              [
                "2026-03-13T22:00:00",
                21
              ],
              [
                "2026-03-13T23:00:00",
                17
              ],
              [
                "2026-03-14T00:00:00",
                12
              ],
              [
                "2026-03-14T01:00:00",
                7
              ],
              [
                "2026-03-14T02:00:00",
                4
              ],
              [
                "2026-03-14T03:00:00",
                4
              ],
              [
                "2026-03-14T04:00:00",
                7
              ],
              [
                "2026-03-14T05:00:00",
                11
              ],
              [
                "2026-03-14T06:00:00",
                21
              ],
              [
                "2026-03-14T07:00:00",
                29
              ],
              [
                "2026-03-14T08:00:00",
                33
              ],
              [
                "2026-03-14T09:00:00",
                38
              ],
              [
                "2026-03-14T10:00:00",
                39
              ],
              [
                "2026-03-14T11:00:00",
                37
              ],
              [
                "2026-03-14T12:00:00",
                31
              ],
              [
                "2026-03-14T13:00:00",
                26
              ],
              [
                "2026-03-14T14:00:00",
                24
              ],
              [
                "2026-03-14T15:00:00",
                20
              ],
              [
                "2026-03-14T16:00:00",
                20
              ],
              [
                "2026-03-14T17:00:00",
                22
              ],
              [
                "2026-03-14T18:00:00",
                24
              ],
              [
                "2026-03-14T19:00:00",
                26
              ],
              [
                "2026-03-14T20:00:00",
                26
              ],
              [
                "2026-03-14T21:00:00",
                27
              ],
              [
                "2026-03-14T22:00:00",
                23
              ],
              [
                "2026-03-14T23:00:00",
                18
              ],
              [
                "2026-03-15T00:00:00",
                13
              ],
              [
                "2026-03-15T01:00:00",
                10
              ],
              [
                "2026-03-15T02:00:00",
                4
              ],
              [
                "2026-03-15T03:00:00",
                3
              ],
              [
                "2026-03-15T04:00:00",
                7
              ],
              [
                "2026-03-15T05:00:00",
                14
              ],
              [
                "2026-03-15T06:00:00",
                20
              ],
              [
                "2026-03-15T07:00:00",
                29
              ],
              [
                "2026-03-15T08:00:00",
                34
              ],
              [
                "2026-03-15T09:00:00",
                39
              ],
              [
                "2026-03-15T10:00:00",
                38
              ],
              [
                "2026-03-15T11:00:00",
                37
              ],
              [
                "2026-03-15T12:00:00",
                32
              ],
              [
                "2026-03-15T13:00:00",
                26
              ],
              [
                "2026-03-15T14:00:00",
                23
              ],
              [
                "2026-03-15T15:00:00",
                21
              ]
            ]
          },
          "Product B": {
            "columns": [
              "timestamp",
              "units_sold"
            ],
            "rows": [
              [
                "2026-03-08T15:00:00",
                44
              ],
              [
                "2026-03-08T16:00:00",
                41
              ],
              [
                "2026-03-08T17:00:00",
                47
              ],
              [
                "2026-03-08T18:00:00",
                37
              ],
              [
                "2026-03-08T19:00:00",
                39
              ],
              [
                "2026-03-08T20:00:00",
                32
              ],
              [
                "2026-03-08T21:00:00",
                22
              ],
              [
                "2026-03-08T22:00:00",
                20
              ],
              [
                "2026-03-08T23:00:00",
                17
              ],
              [
                "2026-03-09T00:00:00",
                11
              ],
              [
                "2026-03-09T01:00:00",
                14
              ],
              [
                "2026-03-09T02:00:00",
                7
              ],
              [
                "2026-03-09T03:00:00",
                3
              ],
              [
                "2026-03-09T04:00:00",
                6
              ],
              [
                "2026-03-09T05:00:00",
                7
              ],
              [
                "2026-03-09T06:00:00",
                11
              ],
              [
                "2026-03-09T07:00:00",
                17
              ],
              [
                "2026-03-09T08:00:00",
                18
              ],
              [
                "2026-03-09T09:00:00",
                21
              ],
              [
                "2026-03-09T10:00:00",
                30
              ],
              [
                "2026-03-09T11:00:00",
                35
              ],
              [
                "2026-03-09T12:00:00",
                38
              ],
              [
                "2026-03-09T13:00:00",
                41
              ],
              [
                "2026-03-09T14:00:00",
                46
              ],
              [
                "2026-03-09T15:00:00",
                47
              ],
              [
                "2026-03-09T16:00:00",
                43
              ],
              [
                "2026-03-09T17:00:00",
                41
              ],
              [
                "2026-03-09T18:00:00",
                39
              ],
              [
                "2026-03-09T19:00:00",
                33
              ],
              [
                "2026-03-09T20:00:00",
                33
              ],
              [
                "2026-03-09T21:00:00",
                26
              ],
              [
                "2026-03-09T22:00:00",
                20
              ],
              [
                "2026-03-09T23:00:00",
                17
              ],
              [
                "2026-03-10T00:00:00",
                7
              ],
              [
                "2026-03-10T01:00:00",
                9
              ],
              [
                "2026-03-10T02:00:00",
                11
              ],
              [
                "2026-03-10T03:00:00",
                7
              ],
              [
                "2026-03-10T04:00:00",
                4
              ],
              [
                "2026-03-10T05:00:00",
                9
              ],
              [
                "2026-03-10T06:00:00",
                11
              ],
              [
                "2026-03-10T07:00:00",
                15
              ],
              [
                "2026-03-10T08:00:00",
                18
              ],
              [
                "2026-03-10T09:00:00",
                24
              ],
              [
                "2026-03-10T10:00:00",
                29
              ],
              [
                "2026-03-10T11:00:00",
                39
              ],
              [
                "2026-03-10T12:00:00",
                42
              ],
              [
                "2026-03-10T13:00:00",
                42
              ],
              [
                "2026-03-10T14:00:00",
                46
              ],
              [
                "2026-03-10T15:00:00",
                42
              ],
              [
                "2026-03-10T16:00:00",
                45
              ],
              [
                "2026-03-10T17:00:00",
                46
              ],
              [
                "2026-03-10T18:00:00",
                39
              ],
              [
                "2026-03-10T19:00:00",
                37
              ],
              [
                "2026-03-10T20:00:00",
                32
              ],
              [
                "2026-03-10T21:00:00",
                27
              ],
              [
                "2026-03-10T22:00:00",
                21
              ],
              [
                "2026-03-10T23:00:00",
                15
              ],
              [
                "2026-03-11T00:00:00",
                12
              ],
              [
                "2026-03-11T01:00:00",
                9
              ],
              [
                "2026-03-11T02:00:00",
                6
              ],
              [
                "2026-03-11T03:00:00",
                7
              ],
              [
                "2026-03-11T04:00:00",
                6
              ],
              [
                "2026-03-11T05:00:00",
                9
              ],
              [
                "2026-03-11T06:00:00",
                12
              ],
              [
                "2026-03-11T07:00:00",
                16
              ],
              [
                "2026-03-11T08:00:00",
                22
              ],
              [
                "2026-03-11T09:00:00",
                26
              ],
              [
                "2026-03-11T10:00:00",
                30
              ],
              [
                "2026-03-11T11:00:00",
                33
              ],
              [
                "2026-03-11T12:00:00",
                34
              ],
              [
                "2026-03-11T13:00:00",
                40
              ],
              [
                "2026-03-11T14:00:00",
                42
              ],
              [
                "2026-03-11T15:00:00",
                44
              ],
              [
                "2026-03-11T16:00:00",
                48
              ],
              [
                "2026-03-11T17:00:00",
                42
              ],
              [
                "2026-03-11T18:00:00",
                39
              ],
              [
                "2026-03-11T19:00:00",
                36
              ],
              [
                "2026-03-11T20:00:00",
                28
              ],
              [
                "2026-03-11T21:00:00",
                27
              ],
              [
                "2026-03-11T22:00:00",
                19
              ],
              [
                "2026-03-11T23:00:00",
                16
              ],
              [
                "2026-03-12T00:00:00",
                8
              ],
              [
                "2026-03-12T01:00:00",
                10
              ],
              [
                "2026-03-12T02:00:00",
                5
              ],
              [
                "2026-03-12T03:00:00",
                3
              ],
              [
                "2026-03-12T04:00:00",
                7
              ],
              [
                "2026-03-12T05:00:00",
                9
              ],
              [
                "2026-03-12T06:00:00",
                9
              ],
              [
                "2026-03-12T07:00:00",
                13
              ],
              [
                "2026-03-12T08:00:00",
                17
              ],
              [
                "2026-03-12T09:00:00",
                24
              ],
              [
                "2026-03-12T10:00:00",
                31
              ],
              [
                "2026-03-12T11:00:00",
                38
              ],
              [
                "2026-03-12T12:00:00",
                36
              ],
              [
                "2026-03-12T13:00:00",
                42
              ],
              [
                "2026-03-12T14:00:00",
                44
              ],
              [
                "2026-03-12T15:00:00",
                46
              ],
              [
                "2026-03-12T16:00:00",
                43
              ],
              [
                "2026-03-12T17:00:00",
                48
              ],
              [
                "2026-03-12T18:00:00",
                39
              ],
              [
                "2026-03-12T19:00:00",
                35
              ],
              [
                "2026-03-12T20:00:00",
                31
              ],
              [
                "2026-03-12T21:00:00",
                24
              ],
              [
                "2026-03-12T22:00:00",
                16
              ],
              [
                "2026-03-12T23:00:00",
                15
              ],
              [
                "2026-03-13T00:00:00",
                15
              ],
              [
                "2026-03-13T01:00:00",
                11
              ],
              [
                "2026-03-13T02:00:00",
                8
              ],
              [
                "2026-03-13T03:00:00",
                7
              ],
              [
                "2026-03-13T04:00:00",
                6
              ],
              [
                "2026-03-13T05:00:00",
                9
              ],
              [
                "2026-03-13T06:00:00",
                9
              ],
              [
                "2026-03-13T07:00:00",
                16
              ],
              [
                "2026-03-13T08:00:00",
                17
              ],
              [
                "2026-03-13T09:00:00",
                27
              ],
              [
                "2026-03-13T10:00:00",
                30
              ],
              [
                "2026-03-13T11:00:00",
                34
              ],
              [
                "2026-03-13T12:00:00",
                40
              ],
              [
                "2026-03-13T13:00:00",
                44
              ],
              [
                "2026-03-13T14:00:00",
                42
              ],
              [
                "2026-03-13T15:00:00",
                48
              ],
              [
                "2026-03-13T16:00:00",
                44
              ],
              [
                "2026-03-13T17:00:00",
                43
              ],
              [
                "2026-03-13T18:00:00",
                39
              ],
              [
                "2026-03-13T19:00:00",
                32
              ],
              [
                "2026-03-13T20:00:00",
                30
              ],
              [
                "2026-03-13T21:00:00",
                23
              ],
              [
                "2026-03-13T22:00:00",
                20
              ],
              [
                "2026-03-13T23:00:00",
                15
              ],
              [
                "2026-03-14T00:00:00",
                9
              ],
              [
                "2026-03-14T01:00:00",
                8
              ],
              [
                "2026-03-14T02:00:00",
                5
              ],
              [
                "2026-03-14T03:00:00",
                7
              ],
              [
                "2026-03-14T04:00:00",
                8
              ],
              [
                "2026-03-14T05:00:00",
                7
              ],
              [
                "2026-03-14T06:00:00",
                13
              ],
              [
                "2026-03-14T07:00:00",
                19
              ],
              [
                "2026-03-14T08:00:00",
                19
              ],
              [
                "2026-03-14T09:00:00",
                25
              ],
              [
                "2026-03-14T10:00:00",
                31
              ],
              [
                "2026-03-14T11:00:00",
                35
              ],
              [
                "2026-03-14T12:00:00",
                40
              ],
              [
                "2026-03-14T13:00:00",
                43
              ],
              [
                "2026-03-14T14:00:00",
                45
              ],
              [
                "2026-03-14T15:00:00",
                44
              ],
              [
                "2026-03-14T16:00:00",
                44
              ],
              [
                "2026-03-14T17:00:00",
                44
              ],
              [
                "2026-03-14T18:00:00",
                42
              ],
              [
                "2026-03-14T19:00:00",
                32
              ],
              [
                "2026-03-14T20:00:00",
                31
              ],
              [
                "2026-03-14T21:00:00",
                26
              ],
              [
                "2026-03-14T22:00:00",
                20
              ],
              [
                "2026-03-14T23:00:00",
                13
              ],
              [
                "2026-03-15T00:00:00",
                10
              ],
              [
                "2026-03-15T01:00:00",
                10
              ],
              [
                "2026-03-15T02:00:00",
                7
              ],
              [
                "2026-03-15T03:00:00",
                8
              ],
              [
                "2026-03-15T04:00:00",
                12
              ],
              [
                "2026-03-15T05:00:00",
                10
              ],
              [
                "2026-03-15T06:00:00",
                13
              ],
              [
                "2026-03-15T07:00:00",
                14
              ],
              [
                "2026-03-15T08:00:00",
                18
              ],
              [
                "2026-03-15T09:00:00",
                24
              ],
              [
                "2026-03-15T10:00:00",
                32
              ],
              [
                "2026-03-15T11:00:00",
                34
              ],
              [
                "2026-03-15T12:00:00",
                39
              ],
              [
                "2026-03-15T13:00:00",
                39
              ],
              [
                "2026-03-15T14:00:00",
                41
              ],
              [
                "2026-03-15T15:00:00",
                47
              ]
            ]
          },
          "Product C": {
            "columns": [
              "timestamp",
              "units_sold"
            ],
            "rows": [
              [
                "2026-03-08T15:00:00",
                67
              ],
              [
                "2026-03-08T16:00:00",
                65
              ],
              [
                "2026-03-08T17:00:00",
                66
              ],
              [
                "2026-03-08T18:00:00",
                64
              ],
              [
                "2026-03-08T19:00:00",
                51
              ],
              [
                "2026-03-08T20:00:00",
                51
              ],
              [
                "2026-03-08T21:00:00",
                42
              ],
              [
                "2026-03-08T22:00:00",
                30
              ],
              [
                "2026-03-08T23:00:00",
                18
              ],
              [
                "2026-03-09T00:00:00",
                11
              ],
              [
                "2026-03-09T01:00:00",
                16
              ],
              [
                "2026-03-09T02:00:00",
                9
              ],
              [
                "2026-03-09T03:00:00",
                1
              ],
              [
                "2026-03-09T04:00:00",
                13
              ],
              [
                "2026-03-09T05:00:00",
                4
              ],
              [
                "2026-03-09T06:00:00",
                19
              ],
              [
                "2026-03-09T07:00:00",
                24
              ],
              [
                "2026-03-09T08:00:00",
                33
              ],
              [
                "2026-03-09T09:00:00",
                35
              ],
              [
                "2026-03-09T10:00:00",
                39
              ],
              [
                "2026-03-09T11:00:00",
                52
              ],
              [
                "2026-03-09T12:00:00",
                59
              ],
              [
                "2026-03-09T13:00:00",
                65
              ],
              [
                "2026-03-09T14:00:00",
                70
              ],
              [
                "2026-03-09T15:00:00",
                67
              ],
              [
                "2026-03-09T16:00:00",
                67
              ],
              [
                "2026-03-09T17:00:00",
                63
              ],
              [
                "2026-03-09T18:00:00",
                58
              ],
              [
                "2026-03-09T19:00:00",
                53
              ],
              [
                "2026-03-09T20:00:00",
                45
              ],
              [
                "2026-03-09T21:00:00",
                37
              ],
              [
                "2026-03-09T22:00:00",
                28
              ],
              [
                "2026-03-09T23:00:00",
                24
              ],
              [
                "2026-03-10T00:00:00",
                18
              ],
              [
                "2026-03-10T01:00:00",
                8
              ],
              [
                "2026-03-10T02:00:00",
                11
              ],
              [
                "2026-03-10T03:00:00",
                2
              ],
              [
                "2026-03-10T04:00:00",
                14
              ],
              [
                "2026-03-10T05:00:00",
                13
              ],
              [
                "2026-03-10T06:00:00",
                13
              ],
              [
                "2026-03-10T07:00:00",
                23
              ],
              [
                "2026-03-10T08:00:00",
                30
              ],
              [
                "2026-03-10T09:00:00",
                39
              ],
              [
                "2026-03-10T10:00:00",
                44
              ],
              [
                "2026-03-10T11:00:00",
                50
              ],
              [
                "2026-03-10T12:00:00",
                58
              ],
              [
                "2026-03-10T13:00:00",
                62
              ],
              [
                "2026-03-10T14:00:00",
                65
              ],
              [
                "2026-03-10T15:00:00",
                73
              ],
              [
                "2026-03-10T16:00:00",
                64
              ],
              [
                "2026-03-10T17:00:00",
                63
              ],
              [
                "2026-03-10T18:00:00",
                67
              ],
              [
                "2026-03-10T19:00:00",
                53
              ],
              [
                "2026-03-10T20:00:00",
                44
              ],
              [
                "2026-03-10T21:00:00",
                38
              ],
              [
                "2026-03-10T22:00:00",
                32
              ],
              [
                "2026-03-10T23:00:00",
                20
              ],
              [
                "2026-03-11T00:00:00",
                17
              ],
              [
                "2026-03-11T01:00:00",
                12
              ],
              [
                "2026-03-11T02:00:00",
                9
              ],
              [
                "2026-03-11T03:00:00",
                3
              ],
              [
                "2026-03-11T04:00:00",
                16
              ],
              [
                "2026-03-11T05:00:00",
                13
              ],
              [
                "2026-03-11T06:00:00",
                16
              ],
              [
                "2026-03-11T07:00:00",
                29
              ],
              [
                "2026-03-11T08:00:00",
                28
              ],
              [
                "2026-03-11T09:00:00",
                38
              ],
              [
                "2026-03-11T10:00:00",
                47
              ],
              [
                "2026-03-11T11:00:00",
                46
              ],
              [
                "2026-03-11T12:00:00",
                59
              ],
              [
                "2026-03-11T13:00:00",
                63
              ],
              [
                "2026-03-11T14:00:00",
                66
              ],
              [
                "2026-03-11T15:00:00",
                68
              ],
              [
                "2026-03-11T16:00:00",
                69
              ],
              [
                "2026-03-11T17:00:00",
                62
              ],
              [
                "2026-03-11T18:00:00",
                55
              ],
              [
                "2026-03-11T19:00:00",
                47
              ],
              [
                "2026-03-11T20:00:00",
                47
              ],
              [
                "2026-03-11T21:00:00",
                37
              ],
              [
                "2026-03-11T22:00:00",
                28
              ],
              [
                "2026-03-11T23:00:00",
                23
              ],
              [
                "2026-03-12T00:00:00",
                20
              ],
              [
                "2026-03-12T01:00:00",
                12
              ],
              [
                "2026-03-12T02:00:00",
                8
              ],
              [
                "2026-03-12T03:00:00",
                6
              ],
              [
                "2026-03-12T04:00:00",
                7
              ],
              [
                "2026-03-12T05:00:00",
                13
              ],
              [
                "2026-03-12T06:00:00",
                18
              ],
              [
                "2026-03-12T07:00:00",
                18
              ],
              [
                "2026-03-12T08:00:00",
                26
              ],
              [
                "2026-03-12T09:00:00",
                40
              ],
              [
                "2026-03-12T10:00:00",
                44
              ],
              [
                "2026-03-12T11:00:00",
                52
              ],
              [
                "2026-03-12T12:00:00",
                60
              ],
              [
                "2026-03-12T13:00:00",
                64
              ],
              [
                "2026-03-12T14:00:00",
                65
              ],
              [
                "2026-03-12T15:00:00",
                67
              ],
              [
                "2026-03-12T16:00:00",
                64
              ],
              [
                "2026-03-12T17:00:00",
                68
              ],
              [
                "2026-03-12T18:00:00",
                57
              ],
              [
                "2026-03-12T19:00:00",
                55
              ],
              [
                "2026-03-12T20:00:00",
                50
              ],
              [
                "2026-03-12T21:00:00",
                36
              ],
              [
                "2026-03-12T22:00:00",
                28
              ],
              [
                "2026-03-12T23:00:00",
                21
              ],
              [
                "2026-03-13T00:00:00",
                16
              ],
              [
                "2026-03-13T01:00:00",
                16
              ],
              [
                "2026-03-13T02:00:00",
                13
              ],
              [
                "2026-03-13T03:00:00",
                6
              ],
              [
                "2026-03-13T04:00:00",
                9
              ],
              [
                "2026-03-13T05:00:00",
                14
              ],
              [
                "2026-03-13T06:00:00",
                18
              ],
              [
                "2026-03-13T07:00:00",
                22
              ],
              [
                "2026-03-13T08:00:00",
                26
              ],
              [
                "2026-03-13T09:00:00",
                36
              ],
              [
                "2026-03-13T10:00:00",
                48
              ],
              [
                "2026-03-13T11:00:00",
                53
              ],
              [
                "2026-03-13T12:00:00",
                59
              ],
              [
                "2026-03-13T13:00:00",
                64
              ],
              [
                "2026-03-13T14:00:00",
                70
              ],
              [
                "2026-03-13T15:00:00",
                70
              ],
              [
                "2026-03-13T16:00:00",
                72
              ],
              [
                "2026-03-13T17:00:00",
                62
              ],
              [
                "2026-03-13T18:00:00",
                61
              ],
              [
                "2026-03-13T19:00:00",
                52
              ],
              [
                "2026-03-13T20:00:00",
                41
              ],
              [
                "2026-03-13T21:00:00",
                35
              ],
              [
                "2026-03-13T22:00:00",
                29
              ],
              [
                "2026-03-13T23:00:00",
                22
              ],
              [
                "2026-03-14T00:00:00",
                19
              ],
              [
                "2026-03-14T01:00:00",
                7
              ],
              [
                "2026-03-14T02:00:00",
                9
              ],
              [
                "2026-03-14T03:00:00",
                8
              ],
              [
                "2026-03-14T04:00:00",
                5
              ],
              [
                "2026-03-14T05:00:00",
                19
              ],
              [
                "2026-03-14T06:00:00",
                16
              ],
              [
                "2026-03-14T07:00:00",
                20
              ],
              [
                "2026-03-14T08:00:00",
                25
              ],
              [
                "2026-03-14T09:00:00",
                36
              ],
              [
                "2026-03-14T10:00:00",
                48
              ],
              [
                "2026-03-14T11:00:00",
                55
              ],
              [
                "2026-03-14T12:00:00",
                57
              ],
              [
                "2026-03-14T13:00:00",
                58
              ],
              [
                "2026-03-14T14:00:00",
                68
              ],
              [
                "2026-03-14T15:00:00",
                70
              ],
              [
                "2026-03-14T16:00:00",
                65
              ],
              [
                "2026-03-14T17:00:00",
                61
              ],
              [
                "2026-03-14T18:00:00",
                55
              ],
              [
                "2026-03-14T19:00:00",
                52
              ],
              [
                "2026-03-14T20:00:00",
                41
              ],
              [
                "2026-03-14T21:00:00",
                34
              ],
              [
                "2026-03-14T22:00:00",
                30
              ],
              [
                "2026-03-14T23:00:00",
                24
              ],
              [
                "2026-03-15T00:00:00",
                21
              ],
              [
                "2026-03-15T01:00:00",
                15
              ],
              [
                "2026-03-15T02:00:00",
                5
              ],
              [
                "2026-03-15T03:00:00",
                9
              ],
              [
                "2026-03-15T04:00:00",
                9
              ],
              [
                "2026-03-15T05:00:00",
                14
              ],
              [
                "2026-03-15T06:00:00",
                19
              ],
              [
                "2026-03-15T07:00:00",
                28
              ],
              [
                "2026-03-15T08:00:00",
                31
              ],
              [
                "2026-03-15T09:00:00",
                41
              ],
              [
                "2026-03-15T10:00:00",
                50
              ],
              [
                "2026-03-15T11:00:00",
                51
              ],
              [
                "2026-03-15T12:00:00",
                64
              ],
              [
                "2026-03-15T13:00:00",
                67
              ],
              [
                "2026-03-15T14:00:00",
                69
              ],
              [
                "2026-03-15T15:00:00",
                68
              ]
            ]
          }
        },
        "forecasts": {
          "Product A": {
            "timestamp": [
              "2026-03-15T16:00:00",
              "2026-03-15T17:00:00",
              "2026-03-15T18:00:00",
              "2026-03-15T19:00:00",
              "2026-03-15T20:00:00",
              "2026-03-15T21:00:00",
              "2026-03-15T22:00:00",
              "2026-03-15T23:00:00",
              "2026-03-16T00:00:00",
              "2026-03-16T01:00:00",
              "2026-03-16T02:00:00",
              "2026-03-16T03:00:00",
              "2026-03-16T04:00:00",
              "2026-03-16T05:00:00",
              "2026-03-16T06:00:00",
              "2026-03-16T07:00:00",
              "2026-03-16T08:00:00",
              "2026-03-16T09:00:00",
              "2026-03-16T10:00:00",
              "2026-03-16T11:00:00",
              "2026-03-16T12:00:00",
              "2026-03-16T13:00:00",
              "2026-03-16T14:00:00",
              "2026-03-16T15:00:00",
              "2026-03-16T16:00:00",
              "2026-03-16T17:00:00",
              "2026-03-16T18:00:00",
              "2026-03-16T19:00:00",
              "2026-03-16T20:00:00",
              "2026-03-16T21:00:00",
              "2026-03-16T22:00:00",
              "2026-03-16T23:00:00",
              "2026-03-17T00:00:00",
              "2026-03-17T01:00:00",
              "2026-03-17T02:00:00",
              "2026-03-17T03:00:00",
              "2026-03-17T04:00:00",
              "2026-03-17T05:00:00",
              "2026-03-17T06:00:00",
              "2026-03-17T07:00:00",
              "2026-03-17T08:00:00",
              "2026-03-17T09:00:00",
              "2026-03-17T10:00:00",
              "2026-03-17T11:00:00",
              "2026-03-17T12:00:00",
              "2026-03-17T13:00:00",
              "2026-03-17T14:00:00",
              "2026-03-17T15:00:00"
            ],
            "mean": [
              20.504526138305664,
              21.805526733398438,
              23.69278907775879,
              25.529420852661133,
              26.827110290527344,
              26.324140548706055,
              22.897764205932617,
              17.852794647216797,
              12.321682929992676,
              7.192910194396973,
              3.9130783081054688,
              3.9800643920898438,
              7.332183837890625,
              12.986684799194336,
              20.668590545654297,
              27.952547073364258,
              33.47609329223633,
              38.25192642211914,
              38.400447845458984,
              36.45668029785156,
              31.858444213867188,
              26.570825576782227,
              23.2839412689209,
              20.51091957092285,
              20.30457305908203,
              21.678436279296875,
              23.642099380493164,
              25.584678649902344,
              26.985809326171875,
              26.484193801879883,
              23.02522850036621,
              18.039037704467773,
              12.611462593078613,
              7.602766990661621,
              4.303823471069336,
              4.018426895141602,
              7.423919677734375,
              13.13841438293457,
              20.692811965942383,
              28.037954330444336,
              33.97454071044922,
              38.231040954589844,
              38.78217697143555,
              36.92472457885742,
              32.19001388549805,
              26.802392959594727,
              23.54244041442871,
              20.698514938354492
            ],
            "0.1": [
              19.15822410583496,
              20.50179100036621,
              22.401607513427734,
              24.21921157836914,
              25.45020294189453,
              24.837438583374023,
              21.506221771240234,
              16.50567626953125,
              10.83835220336914,
              5.582372665405273,
              2.285032272338867,
              2.251415252685547,
              5.641073226928711,
              11.404353141784668,
              18.795143127441406,
              25.81307601928711,
              31.14609718322754,
              35.49599838256836,
              35.904144287109375,
              34.05424499511719,
              29.796527862548828,
              24.967273712158203,
              21.782663345336914,
              19.394075393676758,
              19.236570358276367,
              20.604127883911133,
              22.458040237426758,
              24.36791229248047,
              25.4338436126709,
              24.74117088317871,
              21.389162063598633,
              16.495697021484375,
              10.960190773010254,
              5.869045257568359,
              2.5734004974365234,
              2.361164093017578,
              5.705011367797852,
              11.369453430175781,
              18.547435760498047,
              25.60714340209961,
              31.265169143676758,
              35.313446044921875,
              36.011741638183594,
              34.2758674621582,
              29.958742141723633,
              25.103273391723633,
              21.822275161743164,
              19.32248306274414
            ],
            "0.9": [
              21.669395446777344,
              23.053070068359375,
              24.99215316772461,
              26.91657066345215,
              28.313753128051758,
              27.791156768798828,
              24.152969360351562,
              19.0040283203125,
              13.467171669006348,
              8.364714622497559,
              4.990467071533203,
              5.050144195556641,
              8.647828102111816,
              14.336263656616211,
              22.190526962280273,
              29.632858276367188,
              35.39604949951172,
              40.310279846191406,
              40.52391815185547,
              38.57139587402344,
              33.898765563964844,
              28.34296226501465,
              25.04033088684082,
              22.027423858642578,
              21.667360305786133,
              22.99347686767578,
              24.871681213378906,
              26.878807067871094,
              28.497577667236328,
              28.168556213378906,
              24.56938934326172,
              19.346378326416016,
              14.037256240844727,
              8.928860664367676,
              5.4414215087890625,
              5.182031631469727,
              8.82529354095459,
              14.672514915466309,
              22.400375366210938,
              29.78118896484375,
              35.815494537353516,
              40.36121368408203,
              40.88827133178711,
              39.124969482421875,
              34.39842224121094,
              28.938234329223633,
              25.735769271850586,
              22.512235641479492
            ]
          },
          "Product B": {
            "timestamp": [
              "2026-03-15T16:00:00",
              "2026-03-15T17:00:00",
              "2026-03-15T18:00:00",
              "2026-03-15T19:00:00",
              "2026-03-15T20:00:00",
              "2026-03-15T21:00:00",
              "2026-03-15T22:00:00",
              "2026-03-15T23:00:00",
              "2026-03-16T00:00:00",
              "2026-03-16T01:00:00",
              "2026-03-16T02:00:00",
              "2026-03-16T03:00:00",
              "2026-03-16T04:00:00",
              "2026-03-16T05:00:00",
              "2026-03-16T06:00:00",
              "2026-03-16T07:00:00",
              "2026-03-16T08:00:00",
              "2026-03-16T09:00:00",
              "2026-03-16T10:00:00",
              "2026-03-16T11:00:00",
              "2026-03-16T12:00:00",
              "2026-03-16T13:00:00",
              "2026-03-16T14:00:00",
              "2026-03-16T15:00:00",
              "2026-03-16T16:00:00",
              "2026-03-16T17:00:00",
              "2026-03-16T18:00:00",
              "2026-03-16T19:00:00",
              "2026-03-16T20:00:00",
              "2026-03-16T21:00:00",
              "2026-03-16T22:00:00",
              "2026-03-16T23:00:00",
              "2026-03-17T00:00:00",
              "2026-03-17T01:00:00",
              "2026-03-17T02:00:00",
              "2026-03-17T03:00:00",
              "2026-03-17T04:00:00",
              "2026-03-17T05:00:00",
              "2026-03-17T06:00:00",
              "2026-03-17T07:00:00",
              "2026-03-17T08:00:00",
              "2026-03-17T09:00:00",
              "2026-03-17T10:00:00",
              "2026-03-17T11:00:00",
              "2026-03-17T12:00:00",
              "2026-03-17T13:00:00",
              "2026-03-17T14:00:00",
              "2026-03-17T15:00:00"
            ],
            "mean": [
              45.744361877441406,
              44.47789764404297,
              39.96786880493164,
              34.93931579589844,
              30.40752410888672,
              25.078933715820312,
              19.764469146728516,
              15.15910816192627,
              11.105031967163086,
              9.184202194213867,
              7.405527114868164,
              6.835023880004883,
              7.098142623901367,
              8.534454345703125,
              11.933398246765137,
              16.158313751220703,
              19.992860794067383,
              25.135028839111328,
              30.816650390625,
              35.86872863769531,
              39.165767669677734,
              42.46210479736328,
              44.626007080078125,
              45.996063232421875,
              45.7534065246582,
              44.59668731689453,
              40.11052703857422,
              35.24456787109375,
              31.033628463745117,
              25.713680267333984,
              20.289962768554688,
              15.610541343688965,
              11.549332618713379,
              9.544018745422363,
              7.7064056396484375,
              6.958499908447266,
              7.410867691040039,
              9.032258987426758,
              12.31091022491455,
              16.357358932495117,
              20.261585235595703,
              25.50418472290039,
              31.366159439086914,
              36.46283721923828,
              39.60572052001953,
              42.69723129272461,
              44.79161834716797,
              46.19174575805664
            ],
            "0.1": [
              43.202728271484375,
              41.657466888427734,
              37.34388732910156,
              32.49301528930664,
              27.895044326782227,
              22.69713592529297,
              17.48502540588379,
              12.914217948913574,
              8.867912292480469,
              6.805437088012695,
              5.133642196655273,
              4.6055755615234375,
              4.980224609375,
              6.566743850708008,
              9.907864570617676,
              14.042644500732422,
              17.96416664123535,
              23.004030227661133,
              28.466156005859375,
              33.398677825927734,
              36.72842788696289,
              39.793521881103516,
              41.68659973144531,
              42.83111572265625,
              42.48583984375,
              41.06452941894531,
              36.83848571777344,
              32.41840744018555,
              28.241992950439453,
              23.26666259765625,
              18.005842208862305,
              13.322606086730957,
              9.245460510253906,
              7.142238616943359,
              5.440708160400391,
              4.845056533813477,
              5.325952529907227,
              7.058143615722656,
              10.134407043457031,
              14.08029842376709,
              18.03704071044922,
              23.15070152282715,
              28.727794647216797,
              33.606204986572266,
              36.765869140625,
              39.615299224853516,
              41.445343017578125,
              42.517974853515625
            ],
            "0.9": [
              48.60978698730469,
              47.43096923828125,
              42.79112243652344,
              37.6799201965332,
              33.3004150390625,
              27.90454864501953,
              22.537134170532227,
              17.90163230895996,
              13.724783897399902,
              11.975788116455078,
              9.977222442626953,
              9.093873977661133,
              9.463135719299316,
              10.984129905700684,
              14.422269821166992,
              18.565746307373047,
              22.39069175720215,
              27.66008949279785,
              33.55706024169922,
              38.70289611816406,
              42.075965881347656,
              45.412227630615234,
              47.52215576171875,
              48.96488952636719,
              48.8265380859375,
              47.848453521728516,
              43.23004913330078,
              38.305755615234375,
              34.17613983154297,
              28.809988021850586,
              23.35015106201172,
              18.584577560424805,
              14.49343204498291,
              12.689022064208984,
              10.720396995544434,
              9.662089347839355,
              10.225909233093262,
              11.88810920715332,
              15.130329132080078,
              19.02608871459961,
              22.825077056884766,
              28.250431060791016,
              34.244171142578125,
              39.468650817871094,
              42.69873046875,
              45.907142639160156,
              47.94013214111328,
              49.381919860839844
            ]
          },
          "Product C": {
            "timestamp": [
              "2026-03-15T16:00:00",
              "2026-03-15T17:00:00",
              "2026-03-15T18:00:00",
              "2026-03-15T19:00:00",
              "2026-03-15T20:00:00",
              "2026-03-15T21:00:00",
              "2026-03-15T22:00:00",
              "2026-03-15T23:00:00",
              "2026-03-16T00:00:00",
              "2026-03-16T01:00:00",
              "2026-03-16T02:00:00",
              "2026-03-16T03:00:00",
              "2026-03-16T04:00:00",
              "2026-03-16T05:00:00",
              "2026-03-16T06:00:00",
              "2026-03-16T07:00:00",
              "2026-03-16T08:00:00",
              "2026-03-16T09:00:00",
              "2026-03-16T10:00:00",
              "2026-03-16T11:00:00",
              "2026-03-16T12:00:00",
              "2026-03-16T13:00:00",
              "2026-03-16T14:00:00",
              "2026-03-16T15:00:00",
              "2026-03-16T16:00:00",
              "2026-03-16T17:00:00",
              "2026-03-16T18:00:00",
              "2026-03-16T19:00:00",
              "2026-03-16T20:00:00",
              "2026-03-16T21:00:00",
              "2026-03-16T22:00:00",
              "2026-03-16T23:00:00",
              "2026-03-17T00:00:00",
              "2026-03-17T01:00:00",
              "2026-03-17T02:00:00",
              "2026-03-17T03:00:00",
              "2026-03-17T04:00:00",
              "2026-03-17T05:00:00",
              "2026-03-17T06:00:00",
              "2026-03-17T07:00:00",
              "2026-03-17T08:00:00",
              "2026-03-17T09:00:00",
              "2026-03-17T10:00:00",
              "2026-03-17T11:00:00",
              "2026-03-17T12:00:00",
              "2026-03-17T13:00:00",
              "2026-03-17T14:00:00",
              "2026-03-17T15:00:00"
            ],
            "mean": [
              68.25753784179688,
              64.33294677734375,
              58.953407287597656,
              52.182167053222656,
              44.8023567199707,
              36.48902130126953,
              29.293916702270508,
              23.117162704467773,
              18.24557876586914,
              12.591239929199219,
              9.324899673461914,
              7.880804061889648,
              10.169549942016602,
              14.116628646850586,
              18.203784942626953,
              24.026535034179688,
              30.23114776611328,
              38.921058654785156,
              47.02241897583008,
              53.7126350402832,
              60.47807312011719,
              64.84934997558594,
              68.66142272949219,
              70.09188842773438,
              68.77432250976562,
              64.90254974365234,
              59.55329132080078,
              52.759666442871094,
              45.605072021484375,
              37.1990966796875,
              29.969520568847656,
              23.500141143798828,
              18.895471572875977,
              13.273134231567383,
              9.766721725463867,
              7.782873153686523,
              10.10135269165039,
              14.322179794311523,
              18.6790828704834,
              24.71501922607422,
              31.01270866394043,
              39.43730163574219,
              47.36338806152344,
              54.099761962890625,
              61.12646484375,
              65.52592468261719,
              69.14380645751953,
              70.39305114746094
            ],
            "0.1": [
              65.0497055053711,
              61.02140808105469,
              55.667762756347656,
              49.130184173583984,
              41.937583923339844,
              33.930274963378906,
              26.850168228149414,
              20.65131950378418,
              15.420270919799805,
              9.610015869140625,
              6.014139175415039,
              4.563690185546875,
              6.7572479248046875,
              10.61678695678711,
              15.046772003173828,
              20.946298599243164,
              27.35129737854004,
              35.78847122192383,
              43.83382034301758,
              50.54799270629883,
              56.998382568359375,
              61.15869140625,
              64.4871597290039,
              65.62289428710938,
              64.30809020996094,
              60.597747802734375,
              55.28544616699219,
              49.05252456665039,
              42.306427001953125,
              34.51346206665039,
              27.43219757080078,
              21.02750015258789,
              15.994155883789062,
              10.284709930419922,
              6.550397872924805,
              4.792110443115234,
              6.905149459838867,
              10.953042984008789,
              15.455942153930664,
              21.445810317993164,
              27.832324981689453,
              35.966773986816406,
              43.770023345947266,
              50.32490539550781,
              56.82306671142578,
              60.98479080200195,
              64.19062805175781,
              65.11436462402344
            ],
            "0.9": [
              71.67506408691406,
              67.64664459228516,
              62.195289611816406,
              55.31642150878906,
              47.960060119628906,
              39.451725006103516,
              32.17584228515625,
              25.903423309326172,
              21.21776580810547,
              15.803155899047852,
              12.547943115234375,
              10.65243148803711,
              13.38730239868164,
              17.783737182617188,
              21.71316146850586,
              27.42102813720703,
              33.64549255371094,
              42.66716766357422,
              50.950584411621094,
              57.488285064697266,
              64.4905014038086,
              68.61067199707031,
              72.39262390136719,
              73.94755554199219,
              72.77479553222656,
              68.87016296386719,
              63.50397872924805,
              56.57518768310547,
              49.40583038330078,
              40.81800842285156,
              33.40961456298828,
              26.681062698364258,
              22.37085723876953,
              17.016590118408203,
              13.691097259521484,
              11.294649124145508,
              13.996709823608398,
              18.501005172729492,
              22.67669677734375,
              28.56482696533203,
              34.7428092956543,
              43.582984924316406,
              51.486045837402344,
              58.23878479003906,
              65.67634582519531,
              69.92729187011719,
              73.41410827636719,
              74.68563079833984
            ]
          }
        }
      }
    }
    </pre>
    </code>
    <p><strong>Result</strong></p>
    <code>
    <pre class="tool">
    http://localhost:8004/plot_32abcfb962dc412f8909d2f8283c5067.html
    </pre>
    </code>
    </details>

    <p>Here's the combined view for all three products!</p>

    <div style="display: block; margin-right: 1rem">
    <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.4.0.min.js" integrity="sha256-KEmPoupLpFyGMyGAiOsiNDbKDKAvxXAn/W+oQa0ZAfk=" crossorigin="anonymous"></script>                <div id="d4238d02-de07-47eb-b215-6b0b6ece531f" class="plotly-graph-div" style="height:750px; width:100%;"></div>            <script>                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("d4238d02-de07-47eb-b215-6b0b6ece531f")) {                    Plotly.newPlot(                        "d4238d02-de07-47eb-b215-6b0b6ece531f",                        [{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q90%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[21.669395446777344,23.053070068359375,24.99215316772461,26.91657066345215,28.313753128051758,27.791156768798828,24.152969360351562,19.0040283203125,13.467171669006348,8.364714622497559,4.990467071533203,5.050144195556641,8.647828102111816,14.336263656616211,22.190526962280273,29.632858276367188,35.39604949951172,40.310279846191406,40.52391815185547,38.57139587402344,33.898765563964844,28.34296226501465,25.04033088684082,22.027423858642578,21.667360305786133,22.99347686767578,24.871681213378906,26.878807067871094,28.497577667236328,28.168556213378906,24.56938934326172,19.346378326416016,14.037256240844727,8.928860664367676,5.4414215087890625,5.182031631469727,8.82529354095459,14.672514915466309,22.400375366210938,29.78118896484375,35.815494537353516,40.36121368408203,40.88827133178711,39.124969482421875,34.39842224121094,28.938234329223633,25.735769271850586,22.512235641479492],"type":"scatter","xaxis":"x","yaxis":"y"},{"fill":"tonexty","fillcolor":"#c2e5ff","hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q10%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[19.15822410583496,20.50179100036621,22.401607513427734,24.21921157836914,25.45020294189453,24.837438583374023,21.506221771240234,16.50567626953125,10.83835220336914,5.582372665405273,2.285032272338867,2.251415252685547,5.641073226928711,11.404353141784668,18.795143127441406,25.81307601928711,31.14609718322754,35.49599838256836,35.904144287109375,34.05424499511719,29.796527862548828,24.967273712158203,21.782663345336914,19.394075393676758,19.236570358276367,20.604127883911133,22.458040237426758,24.36791229248047,25.4338436126709,24.74117088317871,21.389162063598633,16.495697021484375,10.960190773010254,5.869045257568359,2.5734004974365234,2.361164093017578,5.705011367797852,11.369453430175781,18.547435760498047,25.60714340209961,31.265169143676758,35.313446044921875,36.011741638183594,34.2758674621582,29.958742141723633,25.103273391723633,21.822275161743164,19.32248306274414],"type":"scatter","xaxis":"x","yaxis":"y"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#0588f0","dash":"dot","width":1},"mode":"lines","name":"Predicted Mean","showlegend":true,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[20.504526138305664,21.805526733398438,23.69278907775879,25.529420852661133,26.827110290527344,26.324140548706055,22.897764205932617,17.852794647216797,12.321682929992676,7.192910194396973,3.9130783081054688,3.9800643920898438,7.332183837890625,12.986684799194336,20.668590545654297,27.952547073364258,33.47609329223633,38.25192642211914,38.400447845458984,36.45668029785156,31.858444213867188,26.570825576782227,23.2839412689209,20.51091957092285,20.30457305908203,21.678436279296875,23.642099380493164,25.584678649902344,26.985809326171875,26.484193801879883,23.02522850036621,18.039037704467773,12.611462593078613,7.602766990661621,4.303823471069336,4.018426895141602,7.423919677734375,13.13841438293457,20.692811965942383,28.037954330444336,33.97454071044922,38.231040954589844,38.78217697143555,36.92472457885742,32.19001388549805,26.802392959594727,23.54244041442871,20.698514938354492],"type":"scatter","xaxis":"x","yaxis":"y"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#838383","width":1},"mode":"lines","name":"Historical Data","showlegend":true,"x":["2026-03-08T15:00:00","2026-03-08T16:00:00","2026-03-08T17:00:00","2026-03-08T18:00:00","2026-03-08T19:00:00","2026-03-08T20:00:00","2026-03-08T21:00:00","2026-03-08T22:00:00","2026-03-08T23:00:00","2026-03-09T00:00:00","2026-03-09T01:00:00","2026-03-09T02:00:00","2026-03-09T03:00:00","2026-03-09T04:00:00","2026-03-09T05:00:00","2026-03-09T06:00:00","2026-03-09T07:00:00","2026-03-09T08:00:00","2026-03-09T09:00:00","2026-03-09T10:00:00","2026-03-09T11:00:00","2026-03-09T12:00:00","2026-03-09T13:00:00","2026-03-09T14:00:00","2026-03-09T15:00:00","2026-03-09T16:00:00","2026-03-09T17:00:00","2026-03-09T18:00:00","2026-03-09T19:00:00","2026-03-09T20:00:00","2026-03-09T21:00:00","2026-03-09T22:00:00","2026-03-09T23:00:00","2026-03-10T00:00:00","2026-03-10T01:00:00","2026-03-10T02:00:00","2026-03-10T03:00:00","2026-03-10T04:00:00","2026-03-10T05:00:00","2026-03-10T06:00:00","2026-03-10T07:00:00","2026-03-10T08:00:00","2026-03-10T09:00:00","2026-03-10T10:00:00","2026-03-10T11:00:00","2026-03-10T12:00:00","2026-03-10T13:00:00","2026-03-10T14:00:00","2026-03-10T15:00:00","2026-03-10T16:00:00","2026-03-10T17:00:00","2026-03-10T18:00:00","2026-03-10T19:00:00","2026-03-10T20:00:00","2026-03-10T21:00:00","2026-03-10T22:00:00","2026-03-10T23:00:00","2026-03-11T00:00:00","2026-03-11T01:00:00","2026-03-11T02:00:00","2026-03-11T03:00:00","2026-03-11T04:00:00","2026-03-11T05:00:00","2026-03-11T06:00:00","2026-03-11T07:00:00","2026-03-11T08:00:00","2026-03-11T09:00:00","2026-03-11T10:00:00","2026-03-11T11:00:00","2026-03-11T12:00:00","2026-03-11T13:00:00","2026-03-11T14:00:00","2026-03-11T15:00:00","2026-03-11T16:00:00","2026-03-11T17:00:00","2026-03-11T18:00:00","2026-03-11T19:00:00","2026-03-11T20:00:00","2026-03-11T21:00:00","2026-03-11T22:00:00","2026-03-11T23:00:00","2026-03-12T00:00:00","2026-03-12T01:00:00","2026-03-12T02:00:00","2026-03-12T03:00:00","2026-03-12T04:00:00","2026-03-12T05:00:00","2026-03-12T06:00:00","2026-03-12T07:00:00","2026-03-12T08:00:00","2026-03-12T09:00:00","2026-03-12T10:00:00","2026-03-12T11:00:00","2026-03-12T12:00:00","2026-03-12T13:00:00","2026-03-12T14:00:00","2026-03-12T15:00:00","2026-03-12T16:00:00","2026-03-12T17:00:00","2026-03-12T18:00:00","2026-03-12T19:00:00","2026-03-12T20:00:00","2026-03-12T21:00:00","2026-03-12T22:00:00","2026-03-12T23:00:00","2026-03-13T00:00:00","2026-03-13T01:00:00","2026-03-13T02:00:00","2026-03-13T03:00:00","2026-03-13T04:00:00","2026-03-13T05:00:00","2026-03-13T06:00:00","2026-03-13T07:00:00","2026-03-13T08:00:00","2026-03-13T09:00:00","2026-03-13T10:00:00","2026-03-13T11:00:00","2026-03-13T12:00:00","2026-03-13T13:00:00","2026-03-13T14:00:00","2026-03-13T15:00:00","2026-03-13T16:00:00","2026-03-13T17:00:00","2026-03-13T18:00:00","2026-03-13T19:00:00","2026-03-13T20:00:00","2026-03-13T21:00:00","2026-03-13T22:00:00","2026-03-13T23:00:00","2026-03-14T00:00:00","2026-03-14T01:00:00","2026-03-14T02:00:00","2026-03-14T03:00:00","2026-03-14T04:00:00","2026-03-14T05:00:00","2026-03-14T06:00:00","2026-03-14T07:00:00","2026-03-14T08:00:00","2026-03-14T09:00:00","2026-03-14T10:00:00","2026-03-14T11:00:00","2026-03-14T12:00:00","2026-03-14T13:00:00","2026-03-14T14:00:00","2026-03-14T15:00:00","2026-03-14T16:00:00","2026-03-14T17:00:00","2026-03-14T18:00:00","2026-03-14T19:00:00","2026-03-14T20:00:00","2026-03-14T21:00:00","2026-03-14T22:00:00","2026-03-14T23:00:00","2026-03-15T00:00:00","2026-03-15T01:00:00","2026-03-15T02:00:00","2026-03-15T03:00:00","2026-03-15T04:00:00","2026-03-15T05:00:00","2026-03-15T06:00:00","2026-03-15T07:00:00","2026-03-15T08:00:00","2026-03-15T09:00:00","2026-03-15T10:00:00","2026-03-15T11:00:00","2026-03-15T12:00:00","2026-03-15T13:00:00","2026-03-15T14:00:00","2026-03-15T15:00:00"],"y":[21,20,21,25,23,24,26,25,17,13,7,4,2,7,14,22,26,35,37,39,36,32,27,23,20,19,20,22,25,26,27,23,19,12,8,4,4,9,12,20,28,35,36,39,36,34,26,22,19,21,22,23,25,28,27,23,17,11,9,4,3,6,11,22,29,33,38,38,37,32,26,24,21,21,20,24,27,28,27,23,18,11,8,3,4,7,13,19,29,34,41,38,37,31,29,26,21,19,21,22,25,26,25,21,17,15,6,4,6,7,13,21,27,33,38,38,36,33,25,23,21,20,23,23,26,28,25,21,17,12,7,4,4,7,11,21,29,33,38,39,37,31,26,24,20,20,22,24,26,26,27,23,18,13,10,4,3,7,14,20,29,34,39,38,37,32,26,23,21],"type":"scatter","xaxis":"x","yaxis":"y"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q90%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[48.60978698730469,47.43096923828125,42.79112243652344,37.6799201965332,33.3004150390625,27.90454864501953,22.537134170532227,17.90163230895996,13.724783897399902,11.975788116455078,9.977222442626953,9.093873977661133,9.463135719299316,10.984129905700684,14.422269821166992,18.565746307373047,22.39069175720215,27.66008949279785,33.55706024169922,38.70289611816406,42.075965881347656,45.412227630615234,47.52215576171875,48.96488952636719,48.8265380859375,47.848453521728516,43.23004913330078,38.305755615234375,34.17613983154297,28.809988021850586,23.35015106201172,18.584577560424805,14.49343204498291,12.689022064208984,10.720396995544434,9.662089347839355,10.225909233093262,11.88810920715332,15.130329132080078,19.02608871459961,22.825077056884766,28.250431060791016,34.244171142578125,39.468650817871094,42.69873046875,45.907142639160156,47.94013214111328,49.381919860839844],"type":"scatter","xaxis":"x2","yaxis":"y2"},{"fill":"tonexty","fillcolor":"#c2e5ff","hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q10%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[43.202728271484375,41.657466888427734,37.34388732910156,32.49301528930664,27.895044326782227,22.69713592529297,17.48502540588379,12.914217948913574,8.867912292480469,6.805437088012695,5.133642196655273,4.6055755615234375,4.980224609375,6.566743850708008,9.907864570617676,14.042644500732422,17.96416664123535,23.004030227661133,28.466156005859375,33.398677825927734,36.72842788696289,39.793521881103516,41.68659973144531,42.83111572265625,42.48583984375,41.06452941894531,36.83848571777344,32.41840744018555,28.241992950439453,23.26666259765625,18.005842208862305,13.322606086730957,9.245460510253906,7.142238616943359,5.440708160400391,4.845056533813477,5.325952529907227,7.058143615722656,10.134407043457031,14.08029842376709,18.03704071044922,23.15070152282715,28.727794647216797,33.606204986572266,36.765869140625,39.615299224853516,41.445343017578125,42.517974853515625],"type":"scatter","xaxis":"x2","yaxis":"y2"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#0588f0","dash":"dot","width":1},"mode":"lines","name":"Predicted Mean","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[45.744361877441406,44.47789764404297,39.96786880493164,34.93931579589844,30.40752410888672,25.078933715820312,19.764469146728516,15.15910816192627,11.105031967163086,9.184202194213867,7.405527114868164,6.835023880004883,7.098142623901367,8.534454345703125,11.933398246765137,16.158313751220703,19.992860794067383,25.135028839111328,30.816650390625,35.86872863769531,39.165767669677734,42.46210479736328,44.626007080078125,45.996063232421875,45.7534065246582,44.59668731689453,40.11052703857422,35.24456787109375,31.033628463745117,25.713680267333984,20.289962768554688,15.610541343688965,11.549332618713379,9.544018745422363,7.7064056396484375,6.958499908447266,7.410867691040039,9.032258987426758,12.31091022491455,16.357358932495117,20.261585235595703,25.50418472290039,31.366159439086914,36.46283721923828,39.60572052001953,42.69723129272461,44.79161834716797,46.19174575805664],"type":"scatter","xaxis":"x2","yaxis":"y2"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#838383","width":1},"mode":"lines","name":"Historical Data","showlegend":false,"x":["2026-03-08T15:00:00","2026-03-08T16:00:00","2026-03-08T17:00:00","2026-03-08T18:00:00","2026-03-08T19:00:00","2026-03-08T20:00:00","2026-03-08T21:00:00","2026-03-08T22:00:00","2026-03-08T23:00:00","2026-03-09T00:00:00","2026-03-09T01:00:00","2026-03-09T02:00:00","2026-03-09T03:00:00","2026-03-09T04:00:00","2026-03-09T05:00:00","2026-03-09T06:00:00","2026-03-09T07:00:00","2026-03-09T08:00:00","2026-03-09T09:00:00","2026-03-09T10:00:00","2026-03-09T11:00:00","2026-03-09T12:00:00","2026-03-09T13:00:00","2026-03-09T14:00:00","2026-03-09T15:00:00","2026-03-09T16:00:00","2026-03-09T17:00:00","2026-03-09T18:00:00","2026-03-09T19:00:00","2026-03-09T20:00:00","2026-03-09T21:00:00","2026-03-09T22:00:00","2026-03-09T23:00:00","2026-03-10T00:00:00","2026-03-10T01:00:00","2026-03-10T02:00:00","2026-03-10T03:00:00","2026-03-10T04:00:00","2026-03-10T05:00:00","2026-03-10T06:00:00","2026-03-10T07:00:00","2026-03-10T08:00:00","2026-03-10T09:00:00","2026-03-10T10:00:00","2026-03-10T11:00:00","2026-03-10T12:00:00","2026-03-10T13:00:00","2026-03-10T14:00:00","2026-03-10T15:00:00","2026-03-10T16:00:00","2026-03-10T17:00:00","2026-03-10T18:00:00","2026-03-10T19:00:00","2026-03-10T20:00:00","2026-03-10T21:00:00","2026-03-10T22:00:00","2026-03-10T23:00:00","2026-03-11T00:00:00","2026-03-11T01:00:00","2026-03-11T02:00:00","2026-03-11T03:00:00","2026-03-11T04:00:00","2026-03-11T05:00:00","2026-03-11T06:00:00","2026-03-11T07:00:00","2026-03-11T08:00:00","2026-03-11T09:00:00","2026-03-11T10:00:00","2026-03-11T11:00:00","2026-03-11T12:00:00","2026-03-11T13:00:00","2026-03-11T14:00:00","2026-03-11T15:00:00","2026-03-11T16:00:00","2026-03-11T17:00:00","2026-03-11T18:00:00","2026-03-11T19:00:00","2026-03-11T20:00:00","2026-03-11T21:00:00","2026-03-11T22:00:00","2026-03-11T23:00:00","2026-03-12T00:00:00","2026-03-12T01:00:00","2026-03-12T02:00:00","2026-03-12T03:00:00","2026-03-12T04:00:00","2026-03-12T05:00:00","2026-03-12T06:00:00","2026-03-12T07:00:00","2026-03-12T08:00:00","2026-03-12T09:00:00","2026-03-12T10:00:00","2026-03-12T11:00:00","2026-03-12T12:00:00","2026-03-12T13:00:00","2026-03-12T14:00:00","2026-03-12T15:00:00","2026-03-12T16:00:00","2026-03-12T17:00:00","2026-03-12T18:00:00","2026-03-12T19:00:00","2026-03-12T20:00:00","2026-03-12T21:00:00","2026-03-12T22:00:00","2026-03-12T23:00:00","2026-03-13T00:00:00","2026-03-13T01:00:00","2026-03-13T02:00:00","2026-03-13T03:00:00","2026-03-13T04:00:00","2026-03-13T05:00:00","2026-03-13T06:00:00","2026-03-13T07:00:00","2026-03-13T08:00:00","2026-03-13T09:00:00","2026-03-13T10:00:00","2026-03-13T11:00:00","2026-03-13T12:00:00","2026-03-13T13:00:00","2026-03-13T14:00:00","2026-03-13T15:00:00","2026-03-13T16:00:00","2026-03-13T17:00:00","2026-03-13T18:00:00","2026-03-13T19:00:00","2026-03-13T20:00:00","2026-03-13T21:00:00","2026-03-13T22:00:00","2026-03-13T23:00:00","2026-03-14T00:00:00","2026-03-14T01:00:00","2026-03-14T02:00:00","2026-03-14T03:00:00","2026-03-14T04:00:00","2026-03-14T05:00:00","2026-03-14T06:00:00","2026-03-14T07:00:00","2026-03-14T08:00:00","2026-03-14T09:00:00","2026-03-14T10:00:00","2026-03-14T11:00:00","2026-03-14T12:00:00","2026-03-14T13:00:00","2026-03-14T14:00:00","2026-03-14T15:00:00","2026-03-14T16:00:00","2026-03-14T17:00:00","2026-03-14T18:00:00","2026-03-14T19:00:00","2026-03-14T20:00:00","2026-03-14T21:00:00","2026-03-14T22:00:00","2026-03-14T23:00:00","2026-03-15T00:00:00","2026-03-15T01:00:00","2026-03-15T02:00:00","2026-03-15T03:00:00","2026-03-15T04:00:00","2026-03-15T05:00:00","2026-03-15T06:00:00","2026-03-15T07:00:00","2026-03-15T08:00:00","2026-03-15T09:00:00","2026-03-15T10:00:00","2026-03-15T11:00:00","2026-03-15T12:00:00","2026-03-15T13:00:00","2026-03-15T14:00:00","2026-03-15T15:00:00"],"y":[44,41,47,37,39,32,22,20,17,11,14,7,3,6,7,11,17,18,21,30,35,38,41,46,47,43,41,39,33,33,26,20,17,7,9,11,7,4,9,11,15,18,24,29,39,42,42,46,42,45,46,39,37,32,27,21,15,12,9,6,7,6,9,12,16,22,26,30,33,34,40,42,44,48,42,39,36,28,27,19,16,8,10,5,3,7,9,9,13,17,24,31,38,36,42,44,46,43,48,39,35,31,24,16,15,15,11,8,7,6,9,9,16,17,27,30,34,40,44,42,48,44,43,39,32,30,23,20,15,9,8,5,7,8,7,13,19,19,25,31,35,40,43,45,44,44,44,42,32,31,26,20,13,10,10,7,8,12,10,13,14,18,24,32,34,39,39,41,47],"type":"scatter","xaxis":"x2","yaxis":"y2"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q90%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[71.67506408691406,67.64664459228516,62.195289611816406,55.31642150878906,47.960060119628906,39.451725006103516,32.17584228515625,25.903423309326172,21.21776580810547,15.803155899047852,12.547943115234375,10.65243148803711,13.38730239868164,17.783737182617188,21.71316146850586,27.42102813720703,33.64549255371094,42.66716766357422,50.950584411621094,57.488285064697266,64.4905014038086,68.61067199707031,72.39262390136719,73.94755554199219,72.77479553222656,68.87016296386719,63.50397872924805,56.57518768310547,49.40583038330078,40.81800842285156,33.40961456298828,26.681062698364258,22.37085723876953,17.016590118408203,13.691097259521484,11.294649124145508,13.996709823608398,18.501005172729492,22.67669677734375,28.56482696533203,34.7428092956543,43.582984924316406,51.486045837402344,58.23878479003906,65.67634582519531,69.92729187011719,73.41410827636719,74.68563079833984],"type":"scatter","xaxis":"x3","yaxis":"y3"},{"fill":"tonexty","fillcolor":"#c2e5ff","hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#c2e5ff","width":0.5},"mode":"lines","name":"Predicted Q10%","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[65.0497055053711,61.02140808105469,55.667762756347656,49.130184173583984,41.937583923339844,33.930274963378906,26.850168228149414,20.65131950378418,15.420270919799805,9.610015869140625,6.014139175415039,4.563690185546875,6.7572479248046875,10.61678695678711,15.046772003173828,20.946298599243164,27.35129737854004,35.78847122192383,43.83382034301758,50.54799270629883,56.998382568359375,61.15869140625,64.4871597290039,65.62289428710938,64.30809020996094,60.597747802734375,55.28544616699219,49.05252456665039,42.306427001953125,34.51346206665039,27.43219757080078,21.02750015258789,15.994155883789062,10.284709930419922,6.550397872924805,4.792110443115234,6.905149459838867,10.953042984008789,15.455942153930664,21.445810317993164,27.832324981689453,35.966773986816406,43.770023345947266,50.32490539550781,56.82306671142578,60.98479080200195,64.19062805175781,65.11436462402344],"type":"scatter","xaxis":"x3","yaxis":"y3"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#0588f0","dash":"dot","width":1},"mode":"lines","name":"Predicted Mean","showlegend":false,"x":["2026-03-15T16:00:00","2026-03-15T17:00:00","2026-03-15T18:00:00","2026-03-15T19:00:00","2026-03-15T20:00:00","2026-03-15T21:00:00","2026-03-15T22:00:00","2026-03-15T23:00:00","2026-03-16T00:00:00","2026-03-16T01:00:00","2026-03-16T02:00:00","2026-03-16T03:00:00","2026-03-16T04:00:00","2026-03-16T05:00:00","2026-03-16T06:00:00","2026-03-16T07:00:00","2026-03-16T08:00:00","2026-03-16T09:00:00","2026-03-16T10:00:00","2026-03-16T11:00:00","2026-03-16T12:00:00","2026-03-16T13:00:00","2026-03-16T14:00:00","2026-03-16T15:00:00","2026-03-16T16:00:00","2026-03-16T17:00:00","2026-03-16T18:00:00","2026-03-16T19:00:00","2026-03-16T20:00:00","2026-03-16T21:00:00","2026-03-16T22:00:00","2026-03-16T23:00:00","2026-03-17T00:00:00","2026-03-17T01:00:00","2026-03-17T02:00:00","2026-03-17T03:00:00","2026-03-17T04:00:00","2026-03-17T05:00:00","2026-03-17T06:00:00","2026-03-17T07:00:00","2026-03-17T08:00:00","2026-03-17T09:00:00","2026-03-17T10:00:00","2026-03-17T11:00:00","2026-03-17T12:00:00","2026-03-17T13:00:00","2026-03-17T14:00:00","2026-03-17T15:00:00"],"y":[68.25753784179688,64.33294677734375,58.953407287597656,52.182167053222656,44.8023567199707,36.48902130126953,29.293916702270508,23.117162704467773,18.24557876586914,12.591239929199219,9.324899673461914,7.880804061889648,10.169549942016602,14.116628646850586,18.203784942626953,24.026535034179688,30.23114776611328,38.921058654785156,47.02241897583008,53.7126350402832,60.47807312011719,64.84934997558594,68.66142272949219,70.09188842773438,68.77432250976562,64.90254974365234,59.55329132080078,52.759666442871094,45.605072021484375,37.1990966796875,29.969520568847656,23.500141143798828,18.895471572875977,13.273134231567383,9.766721725463867,7.782873153686523,10.10135269165039,14.322179794311523,18.6790828704834,24.71501922607422,31.01270866394043,39.43730163574219,47.36338806152344,54.099761962890625,61.12646484375,65.52592468261719,69.14380645751953,70.39305114746094],"type":"scatter","xaxis":"x3","yaxis":"y3"},{"hovertemplate":"Timestamp: %{x|%d-%m-%Y %H:%M}\u003cbr\u003e%{fullData.name}: %{y:,.0f}\u003cextra\u003e\u003c\u002fextra\u003e","line":{"color":"#838383","width":1},"mode":"lines","name":"Historical Data","showlegend":false,"x":["2026-03-08T15:00:00","2026-03-08T16:00:00","2026-03-08T17:00:00","2026-03-08T18:00:00","2026-03-08T19:00:00","2026-03-08T20:00:00","2026-03-08T21:00:00","2026-03-08T22:00:00","2026-03-08T23:00:00","2026-03-09T00:00:00","2026-03-09T01:00:00","2026-03-09T02:00:00","2026-03-09T03:00:00","2026-03-09T04:00:00","2026-03-09T05:00:00","2026-03-09T06:00:00","2026-03-09T07:00:00","2026-03-09T08:00:00","2026-03-09T09:00:00","2026-03-09T10:00:00","2026-03-09T11:00:00","2026-03-09T12:00:00","2026-03-09T13:00:00","2026-03-09T14:00:00","2026-03-09T15:00:00","2026-03-09T16:00:00","2026-03-09T17:00:00","2026-03-09T18:00:00","2026-03-09T19:00:00","2026-03-09T20:00:00","2026-03-09T21:00:00","2026-03-09T22:00:00","2026-03-09T23:00:00","2026-03-10T00:00:00","2026-03-10T01:00:00","2026-03-10T02:00:00","2026-03-10T03:00:00","2026-03-10T04:00:00","2026-03-10T05:00:00","2026-03-10T06:00:00","2026-03-10T07:00:00","2026-03-10T08:00:00","2026-03-10T09:00:00","2026-03-10T10:00:00","2026-03-10T11:00:00","2026-03-10T12:00:00","2026-03-10T13:00:00","2026-03-10T14:00:00","2026-03-10T15:00:00","2026-03-10T16:00:00","2026-03-10T17:00:00","2026-03-10T18:00:00","2026-03-10T19:00:00","2026-03-10T20:00:00","2026-03-10T21:00:00","2026-03-10T22:00:00","2026-03-10T23:00:00","2026-03-11T00:00:00","2026-03-11T01:00:00","2026-03-11T02:00:00","2026-03-11T03:00:00","2026-03-11T04:00:00","2026-03-11T05:00:00","2026-03-11T06:00:00","2026-03-11T07:00:00","2026-03-11T08:00:00","2026-03-11T09:00:00","2026-03-11T10:00:00","2026-03-11T11:00:00","2026-03-11T12:00:00","2026-03-11T13:00:00","2026-03-11T14:00:00","2026-03-11T15:00:00","2026-03-11T16:00:00","2026-03-11T17:00:00","2026-03-11T18:00:00","2026-03-11T19:00:00","2026-03-11T20:00:00","2026-03-11T21:00:00","2026-03-11T22:00:00","2026-03-11T23:00:00","2026-03-12T00:00:00","2026-03-12T01:00:00","2026-03-12T02:00:00","2026-03-12T03:00:00","2026-03-12T04:00:00","2026-03-12T05:00:00","2026-03-12T06:00:00","2026-03-12T07:00:00","2026-03-12T08:00:00","2026-03-12T09:00:00","2026-03-12T10:00:00","2026-03-12T11:00:00","2026-03-12T12:00:00","2026-03-12T13:00:00","2026-03-12T14:00:00","2026-03-12T15:00:00","2026-03-12T16:00:00","2026-03-12T17:00:00","2026-03-12T18:00:00","2026-03-12T19:00:00","2026-03-12T20:00:00","2026-03-12T21:00:00","2026-03-12T22:00:00","2026-03-12T23:00:00","2026-03-13T00:00:00","2026-03-13T01:00:00","2026-03-13T02:00:00","2026-03-13T03:00:00","2026-03-13T04:00:00","2026-03-13T05:00:00","2026-03-13T06:00:00","2026-03-13T07:00:00","2026-03-13T08:00:00","2026-03-13T09:00:00","2026-03-13T10:00:00","2026-03-13T11:00:00","2026-03-13T12:00:00","2026-03-13T13:00:00","2026-03-13T14:00:00","2026-03-13T15:00:00","2026-03-13T16:00:00","2026-03-13T17:00:00","2026-03-13T18:00:00","2026-03-13T19:00:00","2026-03-13T20:00:00","2026-03-13T21:00:00","2026-03-13T22:00:00","2026-03-13T23:00:00","2026-03-14T00:00:00","2026-03-14T01:00:00","2026-03-14T02:00:00","2026-03-14T03:00:00","2026-03-14T04:00:00","2026-03-14T05:00:00","2026-03-14T06:00:00","2026-03-14T07:00:00","2026-03-14T08:00:00","2026-03-14T09:00:00","2026-03-14T10:00:00","2026-03-14T11:00:00","2026-03-14T12:00:00","2026-03-14T13:00:00","2026-03-14T14:00:00","2026-03-14T15:00:00","2026-03-14T16:00:00","2026-03-14T17:00:00","2026-03-14T18:00:00","2026-03-14T19:00:00","2026-03-14T20:00:00","2026-03-14T21:00:00","2026-03-14T22:00:00","2026-03-14T23:00:00","2026-03-15T00:00:00","2026-03-15T01:00:00","2026-03-15T02:00:00","2026-03-15T03:00:00","2026-03-15T04:00:00","2026-03-15T05:00:00","2026-03-15T06:00:00","2026-03-15T07:00:00","2026-03-15T08:00:00","2026-03-15T09:00:00","2026-03-15T10:00:00","2026-03-15T11:00:00","2026-03-15T12:00:00","2026-03-15T13:00:00","2026-03-15T14:00:00","2026-03-15T15:00:00"],"y":[67,65,66,64,51,51,42,30,18,11,16,9,1,13,4,19,24,33,35,39,52,59,65,70,67,67,63,58,53,45,37,28,24,18,8,11,2,14,13,13,23,30,39,44,50,58,62,65,73,64,63,67,53,44,38,32,20,17,12,9,3,16,13,16,29,28,38,47,46,59,63,66,68,69,62,55,47,47,37,28,23,20,12,8,6,7,13,18,18,26,40,44,52,60,64,65,67,64,68,57,55,50,36,28,21,16,16,13,6,9,14,18,22,26,36,48,53,59,64,70,70,72,62,61,52,41,35,29,22,19,7,9,8,5,19,16,20,25,36,48,55,57,58,68,70,65,61,55,52,41,34,30,24,21,15,5,9,9,14,19,28,31,41,50,51,64,67,69,68],"type":"scatter","xaxis":"x3","yaxis":"y3"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"tickfont":{"color":"#24292f","size":10},"type":"date","tickformat":"%b %d\u003cbr\u003e%Y","tickangle":0,"mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"yaxis":{"anchor":"x","domain":[0.7777777777777778,1.0],"tickfont":{"color":"#24292f","size":10},"tickformat":",.0f","mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"xaxis2":{"anchor":"y2","domain":[0.0,1.0],"tickfont":{"color":"#24292f","size":10},"type":"date","tickformat":"%b %d\u003cbr\u003e%Y","tickangle":0,"mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"yaxis2":{"anchor":"x2","domain":[0.3888888888888889,0.6111111111111112],"tickfont":{"color":"#24292f","size":10},"tickformat":",.0f","mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"xaxis3":{"anchor":"y3","domain":[0.0,1.0],"tickfont":{"color":"#24292f","size":10},"type":"date","tickformat":"%b %d\u003cbr\u003e%Y","tickangle":0,"mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"yaxis3":{"anchor":"x3","domain":[0.0,0.22222222222222224],"tickfont":{"color":"#24292f","size":10},"tickformat":",.0f","mirror":true,"linecolor":"#cecece","gridcolor":"#e8e8e8","gridwidth":0.5},"annotations":[{"font":{"size":14,"color":"#24292f"},"showarrow":false,"text":"Product A","x":0.5,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":14,"color":"#24292f"},"showarrow":false,"text":"Product B","x":0.5,"xanchor":"center","xref":"paper","y":0.6111111111111112,"yanchor":"bottom","yref":"paper"},{"font":{"size":14,"color":"#24292f"},"showarrow":false,"text":"Product C","x":0.5,"xanchor":"center","xref":"paper","y":0.22222222222222224,"yanchor":"bottom","yref":"paper"}],"margin":{"t":50,"b":50,"l":50,"r":50},"hoverlabel":{"namelength":-1},"legend":{"font":{"color":"#24292f","size":12}, "x": 0, "y": 1.1, "orientation": "h"},"height":750,"paper_bgcolor":"white","plot_bgcolor":"white"},                        {"responsive": true}                    )                };            </script>
    </div>

    <p>Here's a summary of the results across all three products:</p>

    <table><thead><tr><th></th><th><strong>Product A</strong></th><th><strong>Product B</strong></th><th><strong>Product C</strong></th></tr></thead><tbody><tr><td><strong>Historical range</strong></td><td>~2–41 units/hr</td><td>~3–48 units/hr</td><td>~1–73 units/hr</td></tr><tr><td><strong>Daily peak timing</strong></td><td>~9–10 AM</td><td>~3–5 PM</td><td>~2–4 PM</td></tr><tr><td><strong>Forecast peak (Mar 16–17)</strong></td><td>~38–41</td><td>~44–46</td><td>~68–70</td></tr><tr><td><strong>Uncertainty band</strong></td><td>Very tight</td><td>Tight</td><td>Moderate</td></tr></tbody></table>

    <p>Key cross-product observations:</p>
    <div style="margin-right:1rem">
    <ul>
    <li>🕐 <strong>Different peak hours:</strong> Product A peaks in the <strong>morning (~9 AM)</strong>, while Products B and C peak in the <strong>afternoon (~3–5 PM)</strong> — suggesting different customer segments or use cases.</li>
    <li>📊 <strong>Volume hierarchy:</strong> Product C sells 2× more than Product B and 1.7× more than Product A at their respective peaks.</li>
    <li>🔄 <strong>Consistent seasonality:</strong> All three products show strong and stable daily cycles, which Chronos has picked up on very confidently.</li>
    </ul>
    </div>

    </div>
    </div>

The assistant consistently followed all instructions and used the correct tools in the right order.
One limitation worth noting is that rather than joining the ``sales`` and ``products`` tables to resolve product names, the assistant
memorized the product IDs from the ``products`` table and used them directly in subsequent queries.
While this approach worked correctly in this case, it is not robust, as the ``products`` table may change over time.

You can download the full code from our `GitHub repository <https://github.com/flaviagiammarino/machine-learning-blog/tree/main/forecasting_assistant/>`__.

References
***************************************************************************************************************

[1] Ansari, A.F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S.S., Arango, S.P., Kapoor, S. and Zschiegner, J., (2024).
Chronos: Learning the language of time series. *arXiv preprint*,
`doi: 10.48550/arXiv.2403.07815 <https://doi.org/10.48550/arXiv.2403.07815>`__.

[2] Ansari, A.F., Turkmen, C., Shchur, O., and Stella, L. (2024).
Fast and accurate zero-shot forecasting with Chronos-Bolt and AutoGluon.
`AWS Blogs - Artificial Intelligence <https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/>`__.
