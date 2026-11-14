"""Verify the ROS RViz plugin in the system interpreter, without opening RViz."""
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def verify_rviz_tools(python):
    help_text = 'Run: bash perception/scripts/build_ros1.sh visualization'
    result = subprocess.run(['rospack','plugins','--attrib=plugin','rviz'],
                            capture_output=True,text=True,check=True,timeout=10)
    manifests = [line.split(maxsplit=1)[1] for line in result.stdout.splitlines()
                 if line.startswith('social_nav_rviz ') and len(line.split(maxsplit=1)) == 2]
    libraries = [Path(prefix)/'libsocial_nav_rviz_tools.so'
                 for prefix in os.environ.get('LD_LIBRARY_PATH','').split(':') if prefix]
    library = next((path for path in libraries if path.is_file()),None)
    try:
        classes = [item for path in manifests for item in ET.parse(path).getroot().iter('class')]
    except (OSError,ET.ParseError) as error:
        raise RuntimeError('RViz plugin manifest unavailable. '+help_text) from error
    if not library or not any(item.get('name')=='social_nav_rviz/SessionPanel' and
            item.get('type')=='social_nav_rviz::SessionPanel' and
            item.get('base_class_type')=='rviz::Panel' for item in classes):
        raise RuntimeError('RViz session panel is missing from the plugin manifest or library. '+help_text)
    code = ('import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); '
            'marker=lib.social_nav_rviz_session_panel_abi; marker.restype=ctypes.c_int; '
            'assert marker()==1, "unsupported session panel ABI"')
    result = subprocess.run([str(python),'-B','-c',code,str(library)],
                            capture_output=True,text=True,timeout=10)
    if result.returncode:
        raise RuntimeError('RViz plugin library is outdated or unloadable. '+help_text)
    return library
