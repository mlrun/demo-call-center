"""Main app entry point for Vizro dashboard."""

# DEFINE IMPORTS
import pandas as pd
from custom_charts import (
    plot_bar_concerns,
    plot_bar_quality,
    plot_bar_upsales,
    plot_box_communication,
    plot_butterfly_upsales_concerns,
    plot_donut_concerns,
    plot_donut_upsales,
    plot_line_calls_over_time,
    plot_map_call_locations,
    plot_radar_quality,
)
from custom_components import Audio, make_tabs_with_title, update_from_selected_row
from dash import html

import vizro.models as vm
from vizro import Vizro
from vizro.figures import kpi_card, kpi_card_reference
from vizro.tables import dash_ag_grid

# DEFINE CONSTANTS
MIN_ROW_HEIGHT = 420
CONCERN_LABELS = ["Concerns Not Addressed", "Concerns Addressed"]


def px(val: int) -> str:
    """Convert integer value to pixel string."""
    return f"{int(val)}px"


# DEFINE DATA
try:
    df = pd.read_csv("/home/mlrun_code/vizro/data.csv")
except FileNotFoundError:
    raise RuntimeError("The data file 'fake_data.csv' was not found.")
df["Call Date"] = pd.to_datetime(df["Call Date"])
df["Upsale Success Reference"] = 0.25
df["Concern Reference"] = 0.50

# DEFINE DASHBOARD
kpi_container = vm.Container(
    layout=vm.Grid(grid=[[0, 1, 2, 3, 4]], row_gap="0px", col_gap="20px"),
    components=[
        vm.Figure(
            figure=kpi_card_reference(
                data_frame=df,
                value_column="Upsale Success",
                reference_column="Upsale Success Reference",
                title="Upsale Success",
                value_format="{value:.0%}",
                reference_format="{delta_relative:+.1%} vs. target",
                icon="more_up",
                agg_func="mean",
            )
        ),
        vm.Figure(
            figure=kpi_card_reference(
                data_frame=df,
                value_column="Concern Addressed",
                reference_column="Concern Reference",
                title="Concerns Addressed",
                value_format="{value:.0%}",
                reference_format="{delta_relative:+.1%} vs. target",
                agg_func="mean",
                icon="recommend",
            )
        ),
        vm.Figure(
            figure=kpi_card(
                data_frame=df,
                agg_func="count",
                value_column="Caller ID",
                title="Number of Calls",
                icon="call",
            )
        ),
        vm.Figure(
            figure=kpi_card(
                data_frame=df,
                agg_func="nunique",
                value_column="Agent ID",
                title="Number of Agents",
                icon="support_agent",
            )
        ),
        vm.Figure(
            figure=kpi_card(
                data_frame=df,
                agg_func="nunique",
                value_column="Caller ID",
                title="Number of Callers",
                icon="person",
            )
        ),
    ],
)

call_summary_container = vm.Container(
    title="Calls Summary",
    layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT), row_gap="0px"),
    components=[
        vm.Container(
            title="",
            layout=vm.Grid(
                grid=[[0], [1]], row_min_height=px(MIN_ROW_HEIGHT // 2), row_gap="0px"
            ),
            components=[
                vm.Graph(
                    title="Calls over time",
                    figure=plot_line_calls_over_time(df),
                ),
                vm.Graph(
                    title="Upsales and Concerns Addressed",
                    figure=plot_butterfly_upsales_concerns(df),
                ),
            ],
            variant="filled",
        ),
        vm.Container(
            title="",
            layout=vm.Grid(
                grid=[[0]], row_min_height=px(MIN_ROW_HEIGHT), row_gap="0px"
            ),
            components=[
                vm.Graph(
                    title="Call Locations",
                    header="Showing actual number of calls per city",
                    figure=plot_map_call_locations(df),
                )
            ],
            variant="filled",
        ),
    ],
)

upsales_container = make_tabs_with_title(
    title="Upsales",
    tabs=[
        vm.Container(
            title="Percentage",
            layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT)),
            components=[
                vm.Graph(
                    title="Average Across Agents",
                    header="Showing percentage of calls",
                    figure=plot_donut_upsales(
                        data_frame=df,
                        group_column="Agent ID",
                        mode="average",
                    ),
                ),
                vm.Graph(
                    title="Per Agent",
                    header="Showing percentage of calls",
                    figure=plot_donut_upsales(
                        data_frame=df,
                        group_column="Agent ID",
                        mode="comparison",
                    ),
                    footer="(The Agent ID is shown inside each donut)",
                ),
            ],
        ),
        vm.Container(
            title="Absolute",
            layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT)),
            components=[
                vm.Graph(
                    title="Average Across Agents",
                    header="Showing actual number of calls",
                    figure=plot_bar_upsales(
                        data_frame=df,
                        group_column="Agent ID",
                        mode="average",
                    ),
                ),
                vm.Graph(
                    title="Per Agent",
                    header="Showing actual number of calls",
                    figure=plot_bar_upsales(
                        data_frame=df,
                        group_column="Agent ID",
                        mode="comparison",
                    ),
                ),
            ],
        ),
    ],
)

concerns_container = make_tabs_with_title(
    title="Concerns",
    tabs=[
        vm.Container(
            title="Percentage",
            layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT)),
            components=[
                vm.Graph(
                    title="Average Across Agents",
                    header="Showing percentage of calls",
                    figure=plot_donut_concerns(
                        data_frame=df,
                        group_column="Agent ID",
                        count_column="Concern Addressed",
                        label_names=CONCERN_LABELS,
                        mode="average",
                    ),
                ),
                vm.Graph(
                    title="Per Agent",
                    header="Showing percentage of calls",
                    figure=plot_donut_concerns(
                        data_frame=df,
                        group_column="Agent ID",
                        count_column="Concern Addressed",
                        label_names=CONCERN_LABELS,
                        mode="comparison",
                    ),
                    footer="(The Agent ID is shown inside each donut)",
                ),
            ],
        ),
        vm.Container(
            title="Absolute",
            layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT)),
            components=[
                vm.Graph(
                    title="Average Across Agents",
                    header="Showing actual number of calls",
                    figure=plot_bar_concerns(
                        data_frame=df,
                        group_column="Agent ID",
                        mode="average",
                    ),
                ),
                vm.Graph(
                    title="Per Agent",
                    header="Showing actual number of calls",
                    figure=plot_bar_concerns(
                        data_frame=df,
                        group_column="Agent ID",
                        mode="comparison",
                    ),
                ),
            ],
        ),
    ],
)

quality_scores_container = make_tabs_with_title(
    title="Quality Scores",
    tabs=[
        vm.Container(
            title="Absolute",
            layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT)),
            components=[
                vm.Graph(
                    title="Average Across Agents",
                    header="Showing actual score",
                    figure=plot_radar_quality(df, "average"),
                ),
                vm.Graph(
                    title="Per Agent",
                    header="Showing actual score",
                    figure=plot_radar_quality(df, "comparison"),
                    footer="(View the tooltips to see the Agent ID)",
                ),
            ],
        ),
        vm.Container(
            title="Comparison",
            layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT)),
            components=[
                vm.Graph(
                    title="Average Across Agents",
                    header="Showing actual score",
                    figure=plot_bar_quality(df, "average"),
                ),
                vm.Graph(
                    title="Per Agent",
                    header="Showing actual score",
                    figure=plot_bar_quality(df, "comparison"),
                ),
            ],
        ),
    ],
)

effective_communication_container = vm.Container(
    title="Effective Communication",
    layout=vm.Grid(grid=[[0, 1]], row_min_height=px(MIN_ROW_HEIGHT)),
    collapsed=False,
    components=[
        vm.Graph(
            title="Average Across Agents",
            header="Showing actual score",
            figure=plot_box_communication(data_frame=df, mode="average"),
        ),
        vm.Graph(
            title="Per Agent",
            header="Showing actual score",
            figure=plot_box_communication(data_frame=df, mode="comparison"),
        ),
    ],
    variant="filled",
)

transcripts_and_audio_container = vm.Container(
    title="Call transcripts",
    layout=vm.Flex(gap="40px"),
    components=[
        vm.AgGrid(
            id="outer_grid",
            figure=dash_ag_grid(
                id="inner_grid",
                data_frame=df[
                    [
                        "Agent ID",
                        "Caller ID",
                        "Topic",
                        "Summary",
                        "audio_file",
                        "text_file",
                    ]
                ],
                dashGridOptions={
                    "rowSelection": "single",
                    "suppressRowDeselection": True,
                },
                columnState=[
                    {"colId": "audio_file", "hide": True},
                    {"colId": "text_file", "hide": True},
                ],
                columnSize="responsiveSizeToFit",
            ),
            actions=[
                vm.Action(
                    function=update_from_selected_row(),
                    inputs=["inner_grid.selectedRows"],
                    outputs=["transcript.children", "audio.src"],
                )
            ],
        ),
        vm.Container(
            layout=vm.Grid(grid=[[0, 0, 1]]),
            components=[
                vm.Card(
                    id="transcript",
                    text="Select a row from the above table to see a transcript",
                    extra={"style": {"height": "450px"}},
                ),
                Audio(id="audio"),
            ],
        ),
    ],
)

call_center_summary_page = vm.Page(
    title="Call Center Summary",
    layout=vm.Flex(gap="20px"),
    components=[
        kpi_container,
        call_summary_container,
        upsales_container,
        concerns_container,
        quality_scores_container,
        effective_communication_container,
    ],
    controls=[
        vm.Filter(column="Agent ID", selector=vm.Dropdown(title="Agent ID")),
        vm.Filter(column="Caller ID", selector=vm.Dropdown(title="Caller ID")),
        vm.Filter(column="Client Tone"),
        vm.Filter(
            column="Effective Communication",
            selector=vm.RangeSlider(title="Effective Communication Score", step=1),
        ),
        vm.Filter(column="Caller City", selector=vm.Dropdown(title="Caller City")),
    ],
)

call_transcripts_page = vm.Page(
    title="Call Transcripts",
    components=[transcripts_and_audio_container],
    controls=[
        vm.Filter(column="Agent ID", selector=vm.Dropdown(title="Agent ID")),
        vm.Filter(column="Caller ID", selector=vm.Dropdown(title="Caller ID")),
    ],
)

dashboard = vm.Dashboard(pages=[call_center_summary_page, call_transcripts_page])

app = Vizro().build(dashboard)

if __name__ == "__main__":
    app.run()
