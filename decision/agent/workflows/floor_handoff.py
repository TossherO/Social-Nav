"""Supervise an observation-only runtime replacement for a floor transition."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from decision.agent import ROOT
from decision.agent.adapters.manual import ManualNavigation
from decision.agent.adapters.runtime import ManualRuntimeClient


def observation_owner(socket_path, *, allow_missing=False):
    """Identify only the same-user interactive process that owns this session."""
    socket_path=Path(socket_path).resolve()
    if socket_path.name!='commands.sock' or socket_path.parent.name!='run':
        raise ValueError('Expected an interactive observation socket')
    directory=socket_path.parent.parent
    record=json.loads((directory/'session.json').read_text())
    pid=record.get('owner_pid')
    if (record.get('source')!='live' or record.get('execute_requested') is not False or
            record.get('agent_observations') is not True or type(pid) is not int or pid<=1):
        raise ValueError('The prior session is not an owned observation-only Agent runtime')
    proc=Path('/proc')/str(pid)
    try:
        owner=proc.stat().st_uid
        status=(proc/'status').read_text()
        if allow_missing and any(line.startswith('State:') and '\tZ' in line for line in status.splitlines()):
            return None
        args=(proc/'cmdline').read_bytes().decode('utf-8').split('\0')
    except FileNotFoundError:
        if allow_missing:return None
        raise ValueError('Observation owner has already stopped') from None
    if owner!=os.getuid():raise ValueError('Observation owner belongs to another user')
    if 'runtime.interactive' not in args or '--output' not in args:
        raise ValueError('Observation owner is not the expected interactive process')
    i=args.index('--output')
    if i+1>=len(args) or (ROOT/args[i+1]).resolve()!=directory:
        raise ValueError('Observation owner output does not match the selected socket')
    return pid


class FloorHandoff:
    def __init__(self, *, session, transition, target_catalog, config_path, seed, output, source_socket):
        if transition.status!='awaiting_destination_localization':
            raise ValueError('The Agent has not observed elevator exit')
        if session.skill.record['handle'] is not None:
            raise ValueError('The source-floor navigation call is still active')
        stage=session.task_goals.current(transition.goal_id)['requirement']
        if (transition.task_id!=session.state['task_id'] or stage.get('stage_type')!='elevator' or
                stage.get('floor_id')!=transition.from_floor or stage.get('to_floor')!=transition.to_floor or
                session.catalog.value.get('floor_id')!=transition.from_floor or
                target_catalog.value.get('floor_id')!=transition.to_floor or
                target_catalog.value.get('localization_map_id')!=transition.destination_map_id):
            raise ValueError('Floor handoff does not match the active task and catalogs')
        transform=seed.get('floor_transform') or {}
        if (seed.get('verified_localization') is not False or seed.get('floor_id')!=target_catalog.value.get('floor_id') or
                transform.get('from_floor')!=transition.from_floor or transform.get('to_floor')!=transition.to_floor or
                transform.get('to_map_id')!=transition.destination_map_id):
            raise ValueError('Expected a provisional seed for the destination floor')
        self.session,self.transition,self.target_catalog=session,transition,target_catalog
        self.config=Path(config_path).resolve();self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        identity=uuid.uuid4().hex
        self.seed_path=self.output/('floor_seed_'+identity+'.json')
        self.seed_path.write_text(json.dumps(seed,ensure_ascii=False,allow_nan=False,indent=2))
        self.run_dir=self.output/('floor_observation_'+identity)
        self.log_path=self.output/('floor_observation_'+identity+'.log')
        self.owner_pid=observation_owner(source_socket,allow_missing=True)
        self.process=None;self.log=None;self.status='stopping_source';self.error=None;self.wait_reason=None
        if self.owner_pid is not None:
            try:os.kill(self.owner_pid,signal.SIGINT)
            except ProcessLookupError:self.owner_pid=None

    def _source_stopped(self):
        if self.owner_pid is None:return True
        proc=Path('/proc')/str(self.owner_pid)
        if not proc.exists():return True
        try:
            state=next(line for line in (proc/'status').read_text().splitlines() if line.startswith('State:'))
            return '\tZ' in state
        except (FileNotFoundError,StopIteration):return True

    def tick(self):
        if self.status=='stopping_source' and self._source_stopped():
            # All runtime arguments are passed positionally, not interpolated as shell text.
            args=[sys.executable,'-B','-m','runtime.interactive','--live','--config',str(self.config),
                  '--output',str(self.run_dir),'--agent-observations','--handoff-seed',str(self.seed_path),
                  '--vehicle-profile','configs/wheelchair_vehicle.yaml','--motion-profile','trial',
                  '--environment-update-hz','7.5','--environment-window-m','18','--roll-gravity-deg','6',
                  '--gpu',os.environ.get('GPU_ID','0')]
            try:
                self.log=self.log_path.open('x')
                self.process=subprocess.Popen(['bash','-c',
                    'source perception/scripts/ros1_env.sh && source navigation/commissioning/performance.env && exec taskset -c "$SOCIAL_NAV_COMMISSIONING_CPU_LIST" "$@"',
                    'floor-observer',*args],cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            except OSError as error:
                self.status='failed';self.error='destination_start_failed: '+type(error).__name__
                return self.snapshot()
            self.status='starting_destination'
        if self.status=='starting_destination':
            if self.process.poll() is not None:
                self.status='failed';self.error='destination_observation_exited';return self.snapshot()
            socket_path=self.run_dir/'run/commands.sock'
            if not socket_path.exists():return self.snapshot()
            try:
                navigation=ManualNavigation(self.target_catalog,ManualRuntimeClient(socket_path),automatic_arrival=True)
                state=navigation.tick();mapped=(navigation.value.get('control') or {}).get('agent_mapped_pose') or {}
                binding=state.get('map_binding') or {}
                if (state.get('input_stale') or state.get('origin_pose') is None or
                        state.get('localization_status')!='valid' or mapped.get('source')!='current_localization_and_scene_association'):
                    self.wait_reason='destination_localization_not_ready'
                    return self.snapshot()
                association=dict(binding,local_session_id=state['session_id'],local_generation=state['generation'],
                                 local_frame=state['frame_id'],measurement_stamp_s=mapped.get('association_stamp_s'))
                if self.transition.status=='awaiting_destination_localization':
                    self.transition.accept_localization(association=association,now_s=state['stamp_s'],max_age_s=1.,
                                                        clock_domain=state['clock_domain'])
                self.session.rebind_manual_floor(self.target_catalog,navigation,self.transition)
            except (OSError,EOFError,ValueError,KeyError,TypeError) as error:
                self.wait_reason=type(error).__name__+': '+str(error)[:180]
                return self.snapshot()
            self.wait_reason=None
            self.status='complete'
        return self.snapshot()

    def snapshot(self):
        return dict(status=self.status,error=self.error,wait_reason=self.wait_reason,
                    observation_dir=str(self.run_dir),log=str(self.log_path))

    def close(self):
        if self.log is not None:self.log.close();self.log=None
