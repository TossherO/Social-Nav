import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class AgentModelUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_window(self, **kwargs):
        from decision.agent.manual_ui import ManualWindow

        window = ManualWindow(**kwargs)
        self.addCleanup(self.app.processEvents)
        self.addCleanup(window.close)
        return window

    def test_model_switch_loads_supported_efforts_and_stage_defaults(self):
        from decision.agent.core.model_settings import DEFAULT_MODEL, GPT_MODEL, MODEL_EFFORTS

        window = self.make_window()
        combos = (window.planning_thinking, window.execution_thinking, window.monitor_thinking)
        self.assertEqual(window.model.currentData(), DEFAULT_MODEL)
        for model, defaults in ((DEFAULT_MODEL, ('low', 'off', 'off')),
                                (GPT_MODEL, ('low', 'low', 'low')),
                                (DEFAULT_MODEL, ('low', 'off', 'off'))):
            with self.subTest(model=model):
                window.model.setCurrentIndex(window.model.findData(model))
                self.assertEqual(tuple(combo.currentData() for combo in combos), defaults)
                for combo in combos:
                    self.assertEqual(tuple(combo.itemData(index) for index in range(combo.count())),
                                     MODEL_EFFORTS[model])

    def test_manual_and_replay_pass_model_and_stage_efforts_to_agent(self):
        from decision.agent.core.model_settings import GPT_MODEL

        with tempfile.TemporaryDirectory() as root:
            socket = Path(root) / 'commands.sock'
            socket.touch()
            catalog = Path(root) / 'catalog.json'
            catalog.write_text('{}')
            for replay, command in ((False, 'manual'), (True, 'shadow')):
                with self.subTest(replay=replay):
                    window = self.make_window(socket=str(socket), catalog=str(catalog), replay=replay)
                    window.instruction.setText('先到门口，再去电梯')
                    window.model.setCurrentIndex(window.model.findData(GPT_MODEL))
                    window.planning_thinking.setCurrentIndex(window.planning_thinking.findData('medium'))
                    window.execution_thinking.setCurrentIndex(window.execution_thinking.findData('high'))
                    with patch.object(window, 'set_reader', return_value=True), patch.object(window.process, 'start') as start:
                        window.start_task()
                    args = start.call_args.args[1]
                    self.assertEqual(args[3], command)
                    for option, value in (('--model', GPT_MODEL), ('--thinking-mode', 'planning'),
                                          ('--planning-thinking-mode', 'medium'), ('--execution-thinking-mode', 'high'),
                                          ('--monitor-thinking-mode', 'low')):
                        self.assertEqual(args[args.index(option) + 1], value)

    def test_reused_registration_and_running_task_lock_expected_controls(self):
        from PyQt5 import QtCore
        from decision.agent.core.model_settings import GPT_MODEL

        window = self.make_window(replay=True)
        controls = (window.model, window.planning_thinking, window.execution_thinking, window.monitor_thinking)
        plan = dict(instruction='测试任务')
        with patch('decision.agent.core.reuse.load_registered_task', return_value=plan):
            window.load_reuse('old/session.sqlite')
        window.model.setCurrentIndex(window.model.findData(GPT_MODEL))
        self.assertEqual([control.isEnabled() for control in controls], [True, False, True, True])
        for process_state in (QtCore.QProcess.Starting, QtCore.QProcess.Running):
            with patch.object(window.process, 'state', return_value=process_state):
                window.refresh_status()
                self.assertFalse(any(control.isEnabled() for control in controls))
        window.refresh_status()
        self.assertEqual([control.isEnabled() for control in controls], [True, False, True, True])
        window.clear_reuse()
        self.assertTrue(all(control.isEnabled() for control in controls))

    def test_auto_follow_applies_cli_presets_before_start(self):
        from PyQt5 import QtCore, QtWidgets
        from decision.agent.core.model_settings import GPT_MODEL
        from decision.agent.manual_ui import ManualWindow, main

        argv = ['manual_ui', '--replay', '--auto-follow', '--reuse-session', 'old/session.sqlite',
                '--runtime-socket', 'commands.sock', '--catalog', 'catalog.json', '--model', GPT_MODEL,
                '--execution-thinking-mode', 'high', '--monitor-thinking-mode', 'xhigh']
        with patch('sys.argv', argv), patch.object(ManualWindow, 'show'), \
             patch.object(ManualWindow, 'load_reuse', autospec=True) as load_reuse, \
             patch.object(ManualWindow, 'start_task', autospec=True) as start_task, \
             patch.object(QtCore.QTimer, 'singleShot', side_effect=lambda _, callback: callback()), \
             patch.object(QtWidgets.QApplication, 'exec_', return_value=0):
            self.assertEqual(main(), 0)
        window = start_task.call_args.args[0]
        self.addCleanup(window.close)
        load_reuse.assert_called_once_with(window, 'old/session.sqlite')
        self.assertEqual(window.model.currentData(), GPT_MODEL)
        self.assertEqual(window.planning_thinking.currentData(), 'low')
        self.assertEqual(window.execution_thinking.currentData(), 'high')
        self.assertEqual(window.monitor_thinking.currentData(), 'xhigh')


if __name__ == '__main__':
    unittest.main()
