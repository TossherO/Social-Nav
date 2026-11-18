"""An old native library must not silently ignore requested loop options."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from perception.mapping_localization.native import NativeRoll


class RollCapabilitiesTests(unittest.TestCase):
    def test_requested_options_require_advertised_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'config.yaml'
            config.write_text('roll:\n  robustLoopClosure: true\n')
            with patch('perception.mapping_localization.native.C.CDLL', return_value=SimpleNamespace()):
                with self.assertRaisesRegex(RuntimeError, 'rebuild'):
                    NativeRoll(config, library='old.so')
            lib = SimpleNamespace(sn_roll_capabilities=MagicMock(return_value=1))
            with patch('perception.mapping_localization.native.C.CDLL', return_value=lib):
                with self.assertRaisesRegex(RuntimeError, 'does not support'):
                    NativeRoll(config, library='partial.so')
            config.write_text('roll:\n  robustLoopClosure: true\n  robustLoopMinGap: 8\n')
            lib.sn_roll_capabilities.return_value = 3
            with patch('perception.mapping_localization.native.C.CDLL', return_value=lib):
                with self.assertRaisesRegex(RuntimeError, 'does not support'):
                    NativeRoll(config, library='older-robust.so')

    def test_supported_options_reach_native_create(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'config.yaml'
            config.write_text('roll:\n  robustLoopClosure: true\n  traceLoopClosure: true\n')
            names = ('sn_roll_error', 'sn_roll_create', 'sn_roll_destroy', 'sn_roll_step', 'sn_roll_save')
            lib = SimpleNamespace(**{name: MagicMock() for name in names})
            lib.sn_roll_capabilities = MagicMock(return_value=3)
            lib.sn_roll_create.return_value = 1
            with patch('perception.mapping_localization.native.C.CDLL', return_value=lib):
                instance = NativeRoll(config, library='new.so')
                lib.sn_roll_create.assert_called_once()
                instance.close()


if __name__ == '__main__':
    unittest.main()
