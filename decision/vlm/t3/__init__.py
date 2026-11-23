"""T3 seating: candidate geometry, input preparation, inference and phase feedback.

Run commands with ``python -m decision.vlm.t3``. Imports stay lightweight so
bag extraction can use the system ROS interpreter without model dependencies.
"""


def table_keepout_circle(table):
    """Map-frame circle carried by reference requests for the registered table."""
    return dict(center_xy_m=list(table['center_xy_m']), radius_m=table['radius_m'])
