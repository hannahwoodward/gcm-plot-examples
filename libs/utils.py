import cartopy.crs as ccrs
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def create_figure(
    shape=(1, 1),
    height=5,
    projection=None,
    width=5,
):
    '''
    Function `create_figure`

    Inputs:
    - `shape`
        description: set the number of rows and columns
            to show on the figure
        default: (1, 1)
        type: array
    - `height`
        description: how tall to make each subfigure
        default: 5
        type: integer
    - `projection`
        description: whether to use a cartopy projection
            on spatial plots. See
            https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html
        default: None
        type: None|ccrs projection
    - `width`
        description: how wide to make each subfigure
        default: 5
        type: integer

    Output:
    - (fig, axs)
        description: the created figure and array of axes.
        (NB you plot on each axis)
    '''
    subplot_kw = {}
    if type(projection) != type(None):
        subplot_kw = {
            'projection': projection
        }

    fig, axs = plt.subplots(
        *shape,
        figsize=(width * shape[1], height * shape[0]),
        subplot_kw=subplot_kw
    )
    
    # Flatten the axes from a multidimensional array into
    # a single array which makes it easier to loop over
    axs = axs.flatten() if shape[0] > 1 or shape[1] > 1 else [axs]

    return fig, axs


def draw_gridlines(ax):
    '''
    Function `draw_gridlines`

    Inputs:
    - `ax`
        description: the matplotlib axes to draw gridlines on
        type: matplotlib.axes._subplots.AxesSubplot

    Output:
    - None
    '''
    ax.gridlines(
        alpha=0.5,
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=[-135, -90, -45, 0, 45, 90, 135],
        ylocs=[-60, -30, 0, 30, 60]
    )

    # Draw grid labels
    gridlabels = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=['bottom', 'geo', 'left'],
        alpha=0,
        xlocs=[-90, 0, 90],
        ylocs=[-60, -30, 0, 30, 60],
        # xlabel_style={'rotation': 45, 'ha':'right'},
    )

    # Force draw to add label artists
    plt.draw()

    # Remove right hand side geo artists (-60, 60)
    for a in gridlabels.geo_label_artists:
        if a.get_position()[0] > 0:
            a.set_visible(False)