import plotly.graph_objects as go
import pandas as pd
# import streamlit as st


def animate_plot(
    history: pd.DataFrame,
    plot_col: str,
    title: str = "None", # type: ignore
    y_dtick: int = None, # type: ignore
    theme: str = "plotly_dark",
    y_range_loss_range: int = None, # type: ignore
    delay: int = 50,
    animate: bool = False,
    streamlit: bool = False,
    show_grid: bool = True,
    auto_size: bool = False,
    autorange: bool = False,
):
    trace1_col = plot_col
    trace2_col = "val_" + plot_col
    history = pd.DataFrame((history.copy()))

    if y_range_loss_range == None:
        y_range_max_range = max(history[trace1_col].max(), history[trace2_col].max())
    else:
        y_range_max_range = y_range_loss_range
    history[trace1_col] = history[trace1_col].apply(lambda x: round(x, 4))
    history[trace2_col] = history[trace2_col].apply(lambda x: round(x, 4))

    trace1 = go.Scatter(
        x=list(range(1, len(history) + 1)),
        y=history[trace1_col],
        name="train_" + trace1_col,
    )
    trace2 = go.Scatter(
        x=list(range(1, len(history) + 1)), y=history[trace2_col], name=trace2_col
    )
    if animate:
        fig = go.Figure(
            data=[trace1, trace2],
            layout=go.Layout(
                xaxis=dict(
                    range=[1, len(history)],
                    autorange=False,
                ),
                yaxis=dict(range=[0, y_range_max_range], autorange=False),
                # * buttons
                showlegend=True,
                hovermode="x unified",
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": delay, "redraw": False},
                                        "fromcurrent": True,
                                        "transition": {
                                            "duration": 50,
                                            "easing": "quadratic-in-out",
                                        },
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
            ),
            # * frames
            frames=[
                dict(
                    data=[
                        dict(
                            type="scatter",
                            x=list(range(len(history)))[: k + 1],
                            y=history[trace1_col][: k + 1],
                        ),
                        dict(
                            type="scatter",
                            x=list(range(len(history + 1)))[: k + 1],
                            y=history[trace2_col][: k + 1],
                        ),
                    ],
                    traces=[0, 1],
                )
                for k in range(2, len(history) + 1)
            ],
        )

    else:
        fig = go.Figure(
            data=[trace1, trace2],
            layout=go.Layout(
                xaxis=dict(
                    range=[1, len(history) + 1],
                    autorange=False,
                ),
                yaxis=dict(range=[0, y_range_max_range], autorange=False),
                # * buttons
                showlegend=True,
                hovermode="x unified",
            ),
        )

    # * change xtick
    if title == None:
        fig.update_layout(
            title=plot_col + " history of model",
            xaxis_title="Epochs",
            yaxis_title=plot_col,
            title_x=0.5,
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title="Epochs",
            yaxis_title=plot_col,
            title_x=0.5,
        )
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=1, dtick=int(len(history) / 10))
    )
    if plot_col == "accuracy":
        fig.update_layout(
            yaxis=dict(tickmode="linear", tick0=0, dtick=0.1, range=[0, 1])
        )
    else:
        if y_dtick == None:
            fig.update_layout(
                yaxis=dict(
                    tickmode="linear",
                    tick0=0,
                    dtick=round(max(history[trace2_col] / 10 + 0.2), 1),
                    range=[0, max(history[trace2_col] + 0.2)],
                )
            )
        else:
            if y_range_loss_range == None:
                fig.update_layout(
                    yaxis=dict(
                        tickmode="linear",
                        tick0=0,
                        dtick=y_dtick,
                        range=[0, max(history[trace2_col] + 0.2)],
                    )
                )
            else:
                fig.update_layout(
                    yaxis=dict(
                        tickmode="linear",
                        tick0=0,
                        dtick=y_dtick,
                        range=[0, +y_range_loss_range + 0.2],
                    )
                )

    if autorange == True:
        fig.update_layout(yaxis=dict(autorange=True))
    fig.update_layout(xaxis=dict(showgrid=show_grid), yaxis=dict(showgrid=show_grid))
    fig.update_layout(template=theme)
    if not auto_size:
        fig.update_layout(
            autosize=False,
            width=800,
            height=500,
        )

    # * hover modes
    # fig.update_xaxes(showspikes=True, spikecolor="green", spikesnap="cursor", spikemode="across")
    # fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=2)
    # fig.update_layout(hovermode="closest")

    # if streamlit:
    #     st.plotly_chart(fig)
    else:
        fig.show()


themes = [
    "ggplot2",
    "seaborn",
    "simple_white",
    "plotly",
    "plotly_white",
    "plotly_dark",
    "presentation",
    "xgridoff",
    "ygridoff",
    "gridon",
    "none",
]
