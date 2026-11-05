"""Shared, read-only scene arrays. Only descriptors/metadata travel in IPC."""
import copy
from perception.environment.layers import GridSnapshot
from perception.environment.scene import SceneInputs
from .buffers import SharedArrayPool


class ScenePool:
    def __init__(self,slots=2,slot_bytes=16*1024*1024):
        self.pool=SharedArrayPool(slots,slot_bytes)

    def publish(self,scene,reader):
        scene.verify();arrays={};description=dict(record=copy.deepcopy(scene.record),grids={},rgb=scene.rgb is not None)
        for name,grid in (('geometry',scene.geometry),('prior_geometry',scene.prior_geometry)):
            if grid is None:continue
            description['grids'][name]=copy.deepcopy(grid.metadata)
            arrays.update((name+'/'+key,value) for key,value in grid.layers.items())
        if scene.rgb is not None:arrays['rgb']=scene.rgb
        description['frame']=self.pool.publish(arrays,readers=[reader])
        return description

    def release(self,description,reader):self.pool.release(description['frame'],reader)
    def close(self):self.pool.close()


def read_scene(reader,description):
    arrays=reader.arrays(description['frame']);grids={}
    for name,metadata in description['grids'].items():
        prefix=name+'/'
        grids[name]=GridSnapshot(metadata,{k[len(prefix):]:v for k,v in arrays.items() if k.startswith(prefix)})
    return SceneInputs(description['record'],grids['geometry'],arrays.get('rgb'),grids.get('prior_geometry')).verify()
