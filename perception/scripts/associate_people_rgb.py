#!/usr/bin/env python3
"""Online RGB association; run with the project interpreter and sourced ROS workspace."""
from pathlib import Path
import os
import sys

sys.path.insert(0,str(Path(os.environ.get('SOCIAL_NAV_ROOT',Path(__file__).resolve().parents[2])).resolve()))
from perception.detection_tracking.ros.rgb_node import main

if __name__=='__main__':main()
