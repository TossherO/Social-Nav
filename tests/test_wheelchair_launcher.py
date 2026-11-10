"""Exercise launcher decisions with fake Python and taskset; never start hardware."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]


class WheelchairLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('scripts/interactive.sh','navigation/commissioning/performance.env'):
            dest=self.root/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest)
        ros=self.root/'perception/scripts/ros1_env.sh';ros.parent.mkdir(parents=True)
        ros.write_text('# fake ROS setup\n')
        fake=self.root/'fake-python'
        fake.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
log=pathlib.Path(os.environ['TEST_LAUNCH_LOG'])
source=sys.stdin.read() if '-' in sys.argv else ''
with log.open('a') as out:
    if 'load_vehicle_profile' in source: out.write('profile\\n')
    elif 'rosgraph' in source: out.write('clock\\n')
    else: out.write('python '+repr(sys.argv[1:])+'\\n')
if 'load_vehicle_profile' in source and os.environ.get('TEST_PROFILE_FAIL'): sys.exit(1)
if 'rosgraph' in source and os.environ.get('TEST_CLOCK_FAIL'): sys.exit(1)
''')
        fake.chmod(0o755)
        taskset=self.root/'taskset'
        taskset.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
with pathlib.Path(os.environ['TEST_LAUNCH_LOG']).open('a') as out: out.write('taskset '+repr(sys.argv[1:])+'\\n')
''')
        taskset.chmod(0o755)
        self.log=self.root/'calls'
        self.env=dict(os.environ,SOCIAL_NAV_PYTHON=str(fake),TEST_LAUNCH_LOG=str(self.log),
                      PATH=str(self.root)+os.pathsep+os.environ['PATH'])
        for key in ('TEST_PROFILE_FAIL','TEST_CLOCK_FAIL'):
            self.env.pop(key,None)

    def run_script(self,script,*args,**env):
        return subprocess.run(['bash',str(self.root/script),*args],cwd=self.root,
            env=dict(self.env,**env),capture_output=True,text=True,timeout=10)

    def calls(self):
        return self.log.read_text() if self.log.exists() else ''

    def test_dry_runs_select_distinct_modes_without_starting_processes(self):
        observe=self.run_script('scripts/interactive.sh','observe','--dry-run')
        execute=self.run_script('scripts/interactive.sh','execute')
        self.assertEqual(observe.returncode,0,observe.stderr)
        self.assertEqual(execute.returncode,0,execute.stderr)
        obs=shlex.split(observe.stdout);exe=shlex.split(execute.stdout)
        self.assertNotIn('--execute',obs);self.assertNotIn('--chassis-port',obs)
        self.assertIn('--execute',exe);self.assertIn('--chassis-port',exe)
        self.assertIn('navigation/commissioning/interactive_observation.yaml',obs)
        self.assertIn('navigation/commissioning/interactive.yaml',exe)
        for flag in ('--motion-profile','--environment-update-hz','--environment-window-m','--vehicle-profile'):
            self.assertEqual(obs[obs.index(flag)+1],exe[exe.index(flag)+1])
        self.assertEqual(self.calls(),'')

    def test_check_uses_same_python_and_does_not_start_runtime(self):
        result=self.run_script('scripts/interactive.sh','execute','--check')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.calls(),'profile\n')

    def test_profile_failure_and_simulated_clock_do_not_start_execution(self):
        result=self.run_script('scripts/interactive.sh','execute','--execute',TEST_PROFILE_FAIL='1')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.calls(),'profile\n')
        self.log.unlink()
        result=self.run_script('scripts/interactive.sh','execute','--execute',TEST_CLOCK_FAIL='1')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.calls(),'profile\nclock\n')

    def test_observation_checks_clock_before_starting(self):
        result=self.run_script('scripts/interactive.sh','observe',TEST_CLOCK_FAIL='1')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.calls(),'clock\n')

    def test_execution_starts_only_after_checks(self):
        result=self.run_script('scripts/interactive.sh','execute','--execute')
        self.assertEqual(result.returncode,0,result.stderr)
        calls=self.calls()
        self.assertTrue(calls.startswith('profile\nclock\ntaskset '),calls)
        self.assertIn("'--execute'",calls)
        self.assertIn(repr(self.env['SOCIAL_NAV_PYTHON']),calls)

    def test_preflight_uses_selected_python(self):
        result=self.run_script('scripts/interactive.sh','preflight')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("'runtime.preflight'",self.calls())

    def test_explicit_python_takes_precedence_over_local_activation(self):
        activate=self.root/'outputs/runtime/environment/activate.sh'
        activate.parent.mkdir(parents=True)
        activate.write_text('export SOCIAL_NAV_PYTHON=/invalid/python\n')
        result=self.run_script('scripts/interactive.sh','execute','--check')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.calls(),'profile\n')

    def test_invalid_mode_never_starts(self):
        for kind,argument in [('observe','--execute'),('execute','--start')]:
            result=self.run_script('scripts/interactive.sh',kind,argument)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(self.calls(),'')

    def test_general_interactive_parameters_are_forwarded(self):
        result=self.run_script('scripts/interactive.sh','--bag','example.bag','--output','example')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("'runtime.interactive'",self.calls())
        self.assertIn("'--bag'",self.calls())
        self.assertNotIn('taskset',self.calls())


if __name__=='__main__':unittest.main()
