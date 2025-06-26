"""Custom charts for Vizro dashboard.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import vizro.plotly.express as px
from vizro.models.types import capture

CONCERN_LABELS = ["Concerns Not Addressed", "Concerns Addressed"]
UPSALE_LABELS = ["Failed Upsales", "No Upsale Attempted", "Successful Upsales"]


@capture("graph")
def plot_donut_concerns(
    data_frame: pd.DataFrame,
    group_column: str,
    count_column: str,
    label_names: list[str],
    mode: str,
) -> go.Figure:
    """Create a donut chart for concerns addressed, by agent or average.

    Args:
        data_frame (pd.DataFrame): Input data containing agent and concern columns.
        group_column (str): Column name for grouping (e.g., agent ID).
        count_column (str): Column name for concern addressed (boolean).
        label_names (list[str]): List of label names for the donut chart.
        mode (str): 'comparison' for agent subplots, 'average' for overall.

    Returns:
        go.Figure: Plotly Figure object representing the donut chart(s).
    """
    if mode == "comparison":

        agent_count = data_frame[group_column].nunique()

        num_rows = math.ceil(agent_count / 4)
        num_cols = 4

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=None,
            horizontal_spacing=0.08,
            vertical_spacing=0.02,
            specs=[[{"type": "pie"}] * num_cols for _ in range(num_rows)],
        )

        agent_list = data_frame[group_column].unique().tolist()

        for i in range(0, len(agent_list)):
            chart_data = data_frame.copy()
            chart_data = chart_data[chart_data[group_column] == agent_list[i]]

            counts = chart_data[count_column].value_counts()
            labels = label_names

            chart_data = pd.DataFrame(
                {
                    "Labels": labels,
                    "Counts": [counts.get(False, 0), counts.get(True, 0)],
                }
            )

            chart_data.sort_values(by="Labels", ascending=True, inplace=True)

            labels = chart_data["Labels"]
            values = chart_data["Counts"]

            color_discrete_map = {
                "Concerns Addressed": "#00b4ff",
                "Concerns Not Addressed": "#ff9222",
            }
            colors = [color_discrete_map[label] for label in labels]

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.6,
                    title=str(agent_list[i]),
                    marker=dict(colors=colors),
                    sort=False,
                    hovertemplate="Category: %{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
                ),
                row=i // num_cols + 1,
                col=i % num_cols + 1,
            )

        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            opacity=0.9,
        )

        fig.update_traces(textinfo="none")

        fig.update_layout(
            margin_t=0, margin_b=0, margin_l=0, margin_r=0, showlegend=False
        )

    if mode == "average":
        chart_data = data_frame.copy()

        counts = chart_data[count_column].value_counts()
        labels = label_names

        chart_data = pd.DataFrame(
            {"Labels": labels, "Counts": [counts.get(False, 0), counts.get(True, 0)]}
        )

        chart_data.sort_values(by="Labels", ascending=True, inplace=True)

        labels = chart_data["Labels"]
        values = chart_data["Counts"]

        color_discrete_map = {
            "Concerns Addressed": "#00b4ff",
            "Concerns Not Addressed": "#ff9222",
        }
        colors = [color_discrete_map[label] for label in labels]

        fig = go.Figure()

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors),
                sort=False,
                hovertemplate="Category: %{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
            )
        )

        fig.update_layout(margin_t=0, margin_b=0, margin_l=0, margin_r=0)
        fig.update_traces(textposition="outside", textinfo="percent", opacity=0.9)

    return fig


@capture("graph")
def plot_donut_upsales(
    data_frame: pd.DataFrame,
    group_column: str,
    mode: str,
) -> go.Figure:
    """Create a donut chart for upsales outcomes, by agent or average.

    Args:
        data_frame (pd.DataFrame): Input data containing agent and upsale columns.
        group_column (str): Column name for grouping (e.g., agent ID).
        mode (str): 'comparison' for agent subplots, 'average' for overall.

    Returns:
        go.Figure: Plotly Figure object representing the donut chart(s).
    """
    color_discrete_map = {
        "Failed Upsales": "#FF9222",
        "No Upsale Attempted": "#3949AB",
        "Successful Upsales": "#00B4FF",
    }

    labels = ["Failed Upsales", "No Upsale Attempted", "Successful Upsales"]

    if mode == "comparison":

        agent_count = data_frame[group_column].nunique()

        num_rows = math.ceil(agent_count / 4)
        num_cols = 4

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=None,
            horizontal_spacing=0.08,
            vertical_spacing=0.02,
            specs=[[{"type": "pie"}] * num_cols for _ in range(num_rows)],
        )

        agent_list = data_frame[group_column].unique().tolist()

        for i in range(0, len(agent_list)):

            chart_data = data_frame.copy()
            upsale_outcomes = chart_data[chart_data[group_column] == agent_list[i]]

            upsale_outcomes = (
                upsale_outcomes.groupby(["Upsale Attempted", "Upsale Success"])
                .size()
                .reset_index(name="counts")
            )

            def categorize(row: pd.Series) -> str:
                """Categorize upsale outcome for a row.

                Args:
                    row (pd.Series): Row of DataFrame with 'Upsale Attempted' and 'Upsale Success'.
                Returns:
                    str: Category label for the upsale outcome.
                """
                if not row["Upsale Attempted"]:
                    return "No Upsale Attempted"
                elif row["Upsale Success"]:
                    return "Successful Upsales"
                else:
                    return "Failed Upsales"

            upsale_outcomes["category"] = upsale_outcomes.apply(categorize, axis=1)

            counts = upsale_outcomes["category"].value_counts()

            chart_data = pd.DataFrame(
                {
                    "Labels": labels,
                    "Counts": [
                        counts.get("Failed Upsales", 0),
                        counts.get("No Upsale Attempted", 0),
                        counts.get("Successful Upsales", 0),
                    ],
                }
            )

            chart_data.sort_values(by="Labels", ascending=True, inplace=True)

            labels = chart_data["Labels"]
            values = chart_data["Counts"]

            colors = [color_discrete_map[label] for label in labels]

            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.6,
                    title=str(agent_list[i]),
                    marker=dict(colors=colors),
                    sort=False,
                    hovertemplate="Category: %{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
                ),
                row=i // num_cols + 1,
                col=i % num_cols + 1,
            )

        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            opacity=0.9,
        )

        fig.update_traces(textinfo="none")

        fig.update_layout(
            margin_t=0, margin_b=0, margin_l=0, margin_r=0, showlegend=False
        )

    if mode == "average":

        upsale_outcomes = data_frame.copy()

        labels = ["Failed Upsales", "No Upsale Attempted", "Successful Upsales"]

        upsale_outcomes = (
            upsale_outcomes.groupby(["Upsale Attempted", "Upsale Success"])
            .size()
            .reset_index(name="counts")
        )

        def categorize(row: pd.Series) -> str:
            """Categorize upsale outcome for a row.

            Args:
                row (pd.Series): Row of DataFrame with 'Upsale Attempted' and 'Upsale Success'.
            Returns:
                str: Category label for the upsale outcome.
            """
            if not row["Upsale Attempted"]:
                return "No Upsale Attempted"
            elif row["Upsale Success"]:
                return "Successful Upsales"
            else:
                return "Failed Upsales"

        upsale_outcomes["category"] = upsale_outcomes.apply(categorize, axis=1)
        category_counts = (
            upsale_outcomes.groupby("category")["counts"].sum().reset_index()
        )

        counts = dict(zip(category_counts["category"], category_counts["counts"]))

        chart_data = pd.DataFrame(
            {
                "Labels": labels,
                "Counts": [
                    counts.get("Failed Upsales", 0),
                    counts.get("No Upsale Attempted", 0),
                    counts.get("Successful Upsales", 0),
                ],
            }
        )

        chart_data.sort_values(by="Labels", ascending=True, inplace=True)

        labels = chart_data["Labels"]
        values = chart_data["Counts"]

        colors = [color_discrete_map[label] for label in labels]

        fig = go.Figure()

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors),
                sort=False,
                hovertemplate="Category: %{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
            )
        )

        fig.update_layout(
            margin_t=0, margin_b=0, margin_l=0, margin_r=0, legend_traceorder="reversed"
        )
        fig.update_traces(textposition="outside", textinfo="percent", opacity=0.9)

    return fig


@capture("graph")
def plot_bar_concerns(
    data_frame: pd.DataFrame,
    group_column: str,
    mode: str,
) -> go.Figure:
    """Create a bar chart for concerns addressed, by agent or average.

    Args:
        data_frame (pd.DataFrame): Input data containing agent and concern columns.
        group_column (str): Column name for grouping (e.g., agent ID).
        mode (str): 'comparison' for agent subplots, 'average' for overall.

    Returns:
        go.Figure: Plotly Figure object representing the bar chart(s).
    """
    color_discrete_map = {
        "Concerns Addressed": "#00b4ff",
        "Concerns Not Addressed": "#ff9222",
    }

    if mode == "comparison":

        data = pd.DataFrame()

        agent_list = data_frame[group_column].unique().tolist()

        for i in range(0, len(agent_list)):

            chart_data = data_frame.copy()
            chart_data = chart_data[chart_data[group_column] == agent_list[i]]
            chart_data["Concern Addressed"] = chart_data["Concern Addressed"].replace(
                {True: "Concerns Addressed", False: "Concerns Not Addressed"}
            )

            outcomes = (
                chart_data.groupby(["Concern Addressed"])
                .size()
                .reset_index(name="counts")
            )

            category_counts = (
                outcomes.groupby("Concern Addressed")["counts"].sum().reset_index()
            )

            category_counts["agent_id"] = i

            data = pd.concat([data, category_counts])

        fig = px.bar(
            data,
            x="agent_id",
            y="counts",
            color="Concern Addressed",
            title="",
            color_discrete_map=color_discrete_map,
            category_orders={
                "category": [
                    "Concerns Not Addressed",
                    "Concerns Addressed",
                ]
            },
        )

        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(0, len(agent_list))),
                ticktext=agent_list,
            ),
            xaxis_title="Agent ID",
            yaxis_title=None,
        )
        fig.update_traces(
            hovertemplate="Category: %{fullData.name}<br>Count: %{y}<extra></extra>"
        )

    if mode == "average":

        chart_data = data_frame.copy()
        chart_data["Concern Addressed"] = chart_data["Concern Addressed"].replace(
            {True: "Concerns Addressed", False: "Concerns Not Addressed"}
        )

        outcomes = (
            chart_data.groupby(["Concern Addressed"]).size().reset_index(name="counts")
        )

        category_counts = (
            outcomes.groupby("Concern Addressed")["counts"].sum().reset_index()
        )
        category_counts["PLACEHOLDER"] = 1

        fig = px.bar(
            category_counts,
            y="PLACEHOLDER",
            x="counts",
            color="Concern Addressed",
            title="",
            orientation="h",
            text="counts",
            color_discrete_map=color_discrete_map,
            category_orders={
                "category": [
                    "Concerns Not Addressed",
                    "Concerns Addressed",
                ]
            },
        )

        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            legend_title=None,
            margin=dict(t=60),
        )

        fig.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            width=0.2,
            hovertemplate="Category: %{fullData.name}<br>Count: %{x}<extra></extra>",
        )

    return fig


@capture("graph")
def plot_bar_upsales(
    data_frame: pd.DataFrame,
    group_column: str,
    mode: str,
) -> go.Figure:
    """Create a bar chart for upsales outcomes, by agent or average.

    Args:
        data_frame (pd.DataFrame): Input data containing agent and upsale columns.
        group_column (str): Column name for grouping (e.g., agent ID).
        mode (str): 'comparison' for agent subplots, 'average' for overall.

    Returns:
        go.Figure: Plotly Figure object representing the bar chart(s).
    """
    color_discrete_map = {
        "Failed Upsales": "#FF9222",
        "No Upsale Attempted": "#3949AB",
        "Successful Upsales": "#00B4FF",
    }

    if mode == "comparison":

        data = pd.DataFrame()

        agent_list = data_frame[group_column].unique().tolist()

        for i in range(0, len(agent_list)):

            chart_data = data_frame.copy()
            chart_data = chart_data[chart_data[group_column] == agent_list[i]]
            upsale_outcomes = (
                chart_data.groupby(["Upsale Attempted", "Upsale Success"])
                .size()
                .reset_index(name="counts")
            )

            def categorize(row: pd.Series) -> str:
                """Categorize upsale outcome for a row.

                Args:
                    row (pd.Series): Row of DataFrame with 'Upsale Attempted' and 'Upsale Success'.
                Returns:
                    str: Category label for the upsale outcome.
                """
                if not row["Upsale Attempted"]:
                    return "No Upsale Attempted"
                elif row["Upsale Success"]:
                    return "Successful Upsales"
                else:
                    return "Failed Upsales"

            upsale_outcomes["category"] = upsale_outcomes.apply(categorize, axis=1)
            category_counts = (
                upsale_outcomes.groupby("category")["counts"].sum().reset_index()
            )
            category_counts["agent_id"] = i

            data = pd.concat([data, category_counts])

        fig = px.bar(
            data,
            x="agent_id",
            y="counts",
            color="category",
            title="",
            color_discrete_map=color_discrete_map,
            category_orders={
                "category": [
                    "Successful Upsales",
                    "No Upsale Attempted",
                    "Failed Upsales",
                ]
            },
        )

        fig.update_traces(
            hovertemplate="Category: %{fullData.name}<br>Count: %{y}<extra></extra>"
        )

    if mode == "average":

        upsale_outcomes = (
            data_frame.groupby(["Upsale Attempted", "Upsale Success"])
            .size()
            .reset_index(name="counts")
        )

        def categorize(row: pd.Series) -> str:
            """Categorize upsale outcome for a row.

            Args:
                row (pd.Series): Row of DataFrame with 'Upsale Attempted' and 'Upsale Success'.
            Returns:
                str: Category label for the upsale outcome.
            """
            if not row["Upsale Attempted"]:
                return "No Upsale Attempted"
            elif row["Upsale Success"]:
                return "Successful Upsales"
            else:
                return "Failed Upsales"

        upsale_outcomes["category"] = upsale_outcomes.apply(categorize, axis=1)
        category_counts = (
            upsale_outcomes.groupby("category")["counts"].sum().reset_index()
        )
        category_counts["PLACEHOLDER"] = 1

        fig = px.bar(
            category_counts,
            y="PLACEHOLDER",
            x="counts",
            color="category",
            title="",
            orientation="h",
            text="counts",
            color_discrete_map=color_discrete_map,
            category_orders={
                "category": [
                    "Successful Upsales",
                    "No Upsale Attempted",
                    "Failed Upsales",
                ]
            },
        )

        fig.update_traces(
            hovertemplate="Category: %{fullData.name}<br>Count: %{y}<extra></extra>",
            textposition="inside",
            insidetextanchor="middle",
            width=0.2,
        )

        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            legend_title=None,
        )

    return fig


@capture("graph")
def plot_radar_quality(
    data_frame: pd.DataFrame,
    mode: str,
) -> go.Figure:
    """Create a radar (polar) chart for agent communication quality metrics.

    Args:
        data_frame (pd.DataFrame): Input data with agent communication metrics.
        mode (str): 'comparison' for agent subplots, 'average' for overall.

    Returns:
        go.Figure: Plotly Figure object representing the radar chart(s).
    """
    data = data_frame.copy()
    melted_df = pd.melt(
        data,
        id_vars=["Agent ID"],
        value_vars=[
            "Empathy",
            "Professionalism",
            "Kindness",
            "Effective Communication",
            "Active Listening",
        ],
        var_name="Communication Metric",
        value_name="Value",
    )

    grouped_avg_df = melted_df.groupby(
        ["Agent ID", "Communication Metric"], as_index=False
    )["Value"].mean()

    if mode == "comparison":

        agent_count = data_frame["Agent ID"].nunique()

        num_rows = math.ceil(agent_count / 4)
        num_cols = 4

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=None,
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
            specs=[[{"type": "polar"}] * num_cols for _ in range(num_rows)],
        )

        agent_list = grouped_avg_df["Agent ID"].unique().tolist()

        for i in range(0, len(agent_list)):
            chart_data = grouped_avg_df.copy()
            chart_data = chart_data[chart_data["Agent ID"] == agent_list[i]]

            fig.add_trace(
                go.Barpolar(
                    r=chart_data["Value"],
                    theta=chart_data["Communication Metric"],
                    marker_color=[
                        "#00B4FF",
                        "#FF9222",
                        "#3949AB",
                        "#FF5267",
                        "#08BDBA",
                        "#FDC935",
                    ],
                    hovertemplate=f"Agent ID: {agent_list[i]}<br>Metric: %{{theta}}<br>Score: %{{r}}<extra></extra>",
                ),
                row=i // num_cols + 1,
                col=i % num_cols + 1,
            )

        for i in range(num_rows * num_cols):
            fig.update_layout(
                **{
                    f"polar{i + 1}": dict(
                        radialaxis=dict(visible=False, showgrid=False),
                        angularaxis=dict(visible=False, showgrid=False),
                        bgcolor="rgba(0, 0, 0, 0)",
                    )
                }
            )

        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
        )
    if mode == "average":

        grouped_avg_df = melted_df.groupby(
            ["Agent ID", "Communication Metric"], as_index=False
        )["Value"].mean()
        grouped_avg_df = grouped_avg_df.groupby(
            ["Communication Metric"], as_index=False
        )["Value"].mean()

        fig = go.Figure()

        fig.add_trace(
            go.Barpolar(
                r=grouped_avg_df["Value"],
                theta=grouped_avg_df["Communication Metric"],
                marker_color=[
                    "#00B4FF",
                    "#FF9222",
                    "#3949AB",
                    "#FF5267",
                    "#08BDBA",
                    "#FDC935",
                ],
                hovertemplate="Metric: %{theta}<br>Score: %{r}<extra></extra>",
            )
        )

        fig.update_layout(
            polar=dict(
                angularaxis=dict(),
                radialaxis=dict(
                    dtick=1,
                    showgrid=False,
                ),
                bgcolor="rgba(0, 0, 0, 0)",
            ),
            showlegend=False,
        )

    return fig


@capture("graph")
def plot_bar_quality(
    data_frame: pd.DataFrame,
    mode: str,
) -> go.Figure:
    """Create a bar chart for agent communication quality metrics.

    Args:
        data_frame (pd.DataFrame): Input data with agent communication metrics.
        mode (str): 'comparison' for agent subplots, 'average' for overall.

    Returns:
        go.Figure: Plotly Figure object representing the bar chart(s).
    """
    data = data_frame.copy()
    melted_df = pd.melt(
        data,
        id_vars=["Agent ID"],
        value_vars=[
            "Empathy",
            "Professionalism",
            "Kindness",
            "Effective Communication",
            "Active Listening",
        ],
        var_name="Communication Metric",
        value_name="Value",
    )

    grouped_avg_df = melted_df.groupby(
        ["Agent ID", "Communication Metric"], as_index=False
    )["Value"].mean()

    if mode == "comparison":
        agent_count = data_frame["Agent ID"].nunique()
        num_rows = math.ceil(agent_count / 4)
        num_cols = 4
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=None,
            horizontal_spacing=0.04,
            vertical_spacing=0.02,
            specs=[[{"type": "xy"}] * num_cols for _ in range(num_rows)],
        )
        agent_list = grouped_avg_df["Agent ID"].unique().tolist()
        colors = ["#00B4FF", "#FF9222", "#3949AB", "#FF5267", "#08BDBA", "#FDC935"]
        for i, agent in enumerate(agent_list):
            chart_data = grouped_avg_df[grouped_avg_df["Agent ID"] == agent]
            for idx, row in chart_data.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row["Communication Metric"], row["Communication Metric"]],
                        y=[0, row["Value"]],
                        mode="lines",
                        line=dict(color=colors[idx % len(colors)], width=3),
                        showlegend=False,
                    ),
                    row=i // num_cols + 1,
                    col=i % num_cols + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[row["Communication Metric"]],
                        y=[row["Value"]],
                        mode="markers",
                        marker=dict(color=colors[idx % len(colors)], size=8),
                        name=row["Communication Metric"] if i == 0 else None,
                        showlegend=(i == 0),
                        hovertemplate=f"Agent ID: {agent}<br>Metric: %{{x}}<br>Score: %{{y}}<extra></extra>",
                    ),
                    row=i // num_cols + 1,
                    col=i % num_cols + 1,
                )
            fig.update_xaxes(
                showgrid=False,
                visible=True,
                showticklabels=False,
                ticks="",
                title=dict(text=str(agent), font=dict(size=10), standoff=2),
                row=i // num_cols + 1,
                col=i % num_cols + 1,
                zeroline=True,
            )
            fig.update_yaxes(
                showgrid=False,
                visible=False,
                zeroline=False,
                row=i // num_cols + 1,
                col=i % num_cols + 1,
            )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(t=10),
        )

    if mode == "average":
        grouped_avg_df = melted_df.groupby(
            ["Agent ID", "Communication Metric"], as_index=False
        )["Value"].mean()
        grouped_avg_df = grouped_avg_df.groupby(
            ["Communication Metric"], as_index=False
        )["Value"].mean()
        fig = go.Figure()
        colors = ["#00B4FF", "#FF9222", "#3949AB", "#FF5267", "#08BDBA", "#FDC935"]
        for idx, row in grouped_avg_df.iterrows():
            fig.add_trace(
                go.Bar(
                    y=[row["Value"]],
                    x=[row["Communication Metric"]],
                    name=row["Communication Metric"],
                    marker=dict(color=colors[idx % len(colors)]),
                    text=[round(row["Value"], 1)],
                    textposition="inside",
                    hovertemplate="Metric: %{x}<br>Score: %{y}<extra></extra>",
                    width=0.6,
                )
            )
        fig.update_layout(
            showlegend=True,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            xaxis=dict(
                showgrid=False,
                visible=True,
                zeroline=True,
                zerolinecolor="rgba(150,150,150,0.7)",
                zerolinewidth=2,
                showticklabels=False,
                ticks="",
            ),
            yaxis=dict(
                showgrid=False,
                visible=False,
            ),
            barmode="group",
        )
    return fig


@capture("graph")
def plot_box_communication(
    data_frame: pd.DataFrame,
    mode: str,
) -> go.Figure:
    """Create a box plot for Effective Communication scores, by agent or average.

    Args:
        data_frame (pd.DataFrame): Input data with agent and communication scores.
        mode (str): 'comparison' for agent subplots, 'average' for overall.

    Returns:
        go.Figure: Plotly Figure object representing the box plot(s).
    """
    data = data_frame[["Agent ID", "Effective Communication"]].copy()
    data["PLACEHOLDER"] = 1
    if mode == "comparison":
        fig = px.box(data, x="Agent ID", y="Effective Communication")
        fig.update_layout(xaxis=dict(tickvals=data["Agent ID"], tickangle=90))
    if mode == "average":
        fig = px.box(
            data, y="PLACEHOLDER", x="Effective Communication", orientation="h"
        )
        fig.update_layout(
            yaxis=dict(range=[0, 2], visible=False), boxmode="group", bargap=0.5
        )
    return fig


@capture("graph")
def plot_map_call_locations(
    data_frame: pd.DataFrame,
) -> go.Figure:
    """Create a map of call locations with bubble size by call count.

    Args:
        data_frame (pd.DataFrame): Input data with city, latitude, longitude, and call info.

    Returns:
        go.Figure: Plotly Figure object representing the map.
    """
    aggregated_df = (
        data_frame.groupby(["Caller City", "latitude", "longitude"])
        .agg(
            Call_Count=("Caller ID", "count"),
            Agent_IDs=("Agent ID", "count"),
            Caller_Count=("Caller ID", "nunique"),
        )
        .reset_index()
    )
    populations = aggregated_df["Call_Count"]
    min_size = 10
    max_size = 50
    sizes = np.interp(
        aggregated_df["Call_Count"],
        (populations.min(), populations.max()),
        (min_size, max_size),
    )
    fig = go.Figure(
        go.Scattergeo(
            lat=aggregated_df["latitude"],
            lon=aggregated_df["longitude"],
            mode="markers",
            marker=dict(
                size=sizes,
                color="#00B4FF",
                opacity=0.6,
                line=dict(width=0),
            ),
            hovertemplate="City: %{text}<br>Calls: %{customdata[0]:,}<br>Agents: %{customdata[1]:,}<br>Callers: %{customdata[2]:,}<extra></extra>",
            customdata=aggregated_df[["Call_Count", "Agent_IDs", "Caller_Count"]],
            text=aggregated_df["Caller City"],
        )
    )
    fig.update_geos(
        visible=False,
        resolution=110,
        scope="usa",
        showcountries=True,
        countrycolor="rgb(150, 150, 150)",
        showsubunits=True,
        subunitcolor="rgb(150, 150, 150)",
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)
    return fig


@capture("graph")
def plot_line_calls_over_time(
    data_frame: pd.DataFrame,
) -> go.Figure:
    """Create a line chart of number of calls per month.

    Args:
        data_frame (pd.DataFrame): Input data with call dates.

    Returns:
        go.Figure: Plotly Figure object representing the line chart.
    """
    calls_per_month = (
        data_frame.groupby(data_frame["Call Date"].dt.to_period("M"))
        .size()
        .reset_index(name="Count")
    )
    calls_per_month["TickLabel"] = calls_per_month["Call Date"].dt.strftime("%b %y")
    calls_per_month["Call Date"] = calls_per_month["Call Date"].dt.strftime("%Y-%m")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=calls_per_month["Call Date"],
            y=calls_per_month["Count"],
            mode="lines+markers+text",
            text=calls_per_month["Count"],
            textposition="top center",
            hovertemplate="Month: %{x}<br>Count: %{y}<extra></extra>",
            marker=dict(size=6, color="#00B4FF"),
            line=dict(color="#00B4FF", width=2),
            showlegend=False,
            cliponaxis=False,
        )
    )
    fig.update_layout(
        showlegend=False,
        title=None,
        yaxis=dict(visible=False),
        xaxis=dict(
            title=None,
            tickangle=90,
            tickmode="array",
            tickvals=calls_per_month["Call Date"],
            ticktext=calls_per_month["TickLabel"],
            tickfont=dict(size=12),
            showgrid=False,
        ),
        margin=dict(t=10, b=60),
    )
    return fig


@capture("graph")
def plot_butterfly_upsales_concerns(
    data_frame: pd.DataFrame,
) -> go.Figure:
    """Create a butterfly chart comparing upsales and concerns addressed percentages per month.

    Args:
        data_frame (pd.DataFrame): Input data with call dates, upsale, and concern columns.

    Returns:
        go.Figure: Plotly Figure object representing the butterfly chart.
    """
    df = data_frame.copy()
    df["Month"] = df["Call Date"].dt.to_period("M")
    df['Upsale Attempted'].fillna(False, inplace=True)
    upsales = (
        df[df["Upsale Attempted"]]
        .groupby("Month")["Upsale Success"]
        .mean()
        .reset_index()
    )
    upsales["Metric"] = "Upsales Success"
    upsales["Value"] = upsales["Upsale Success"] * 100
    concerns = df.groupby("Month")["Concern Addressed"].mean().reset_index()
    concerns["Metric"] = "Concerns Addressed"
    concerns["Value"] = -concerns["Concern Addressed"] * 100
    plot_df = pd.concat(
        [upsales[["Month", "Metric", "Value"]], concerns[["Month", "Metric", "Value"]]]
    )
    plot_df = plot_df.sort_values(["Month", "Metric"])
    plot_df["MonthLabel"] = plot_df["Month"].dt.strftime("%b %y")
    plot_df = plot_df.sort_values("Month")
    pivot_df = plot_df.pivot(
        index=["Month", "MonthLabel"], columns="Metric", values="Value"
    ).reset_index()
    pivot_df = pivot_df.sort_values("Month")
    month_labels = pivot_df["MonthLabel"]
    if 'Upsales Success' not in pivot_df.columns:
        pivot_df['Upsales Success'] = 0
    else:
        pivot_df['Upsales Success'].fillna(value=0, inplace=True)

    
    if 'Concerns Addressed' not in pivot_df.columns:
        pivot_df['Concerns Addressed'] = 0
    else:
        pivot_df['Concerns Addressed'].fillna(value=0, inplace=True)   
    upsales_y = pivot_df["Upsales Success"].fillna(0)
    concerns_y = pivot_df["Concerns Addressed"].fillna(0)
    fig = go.Figure()
    fig.add_traces(
        [
            go.Bar(
                x=month_labels,
                y=upsales_y,
                name="% Upsales Success",
                marker_color="#00B4FF",
                text=[f"{int(round(v))}%" if v != 0 else "" for v in upsales_y],
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(size=2, color="white"),
                textangle=90,
                offsetgroup=1,
                cliponaxis=False,
                width=0.6,
                hovertemplate="Month: %{x}<br>Upsales Success: %{y:.0f}%<extra></extra>",
            ),
            go.Bar(
                x=month_labels,
                y=concerns_y,
                name="% Concerns Addressed",
                marker_color="#FF9222",
                text=[f"{int(round(abs(v)))}%" if v != 0 else "" for v in concerns_y],
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(size=2, color="white"),
                textangle=90,
                offsetgroup=1,
                cliponaxis=False,
                width=0.6,
                hovertemplate="Month: %{x}<br>Concerns Addressed: %{customdata:.0f}%<extra></extra>",
                customdata=[abs(v) for v in concerns_y],
            ),
        ]
    )
    fig.update_layout(
        barmode="relative",
        bargap=0,
        showlegend=False,
        xaxis=dict(
            visible=True,
            showline=False,
            showticklabels=True,
            ticks="",
            showgrid=False,
            zeroline=False,
            tickangle=90,
            tickfont=dict(size=12),
        ),
        yaxis=dict(visible=False),
        margin=dict(t=0, b=0),
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(150,150,150,0.7)")
    return fig
