import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('h3lv_emotion_controller_test')
package.__path__ = [str(ROOT)]
sys.modules.setdefault(package.__name__, package)
controller = importlib.import_module(package.__name__ + '.controller')
emotion = importlib.import_module(package.__name__ + '.music_emotion')


class EmotionControllerGateTests(unittest.TestCase):
    def test_direct_video_start_is_blocked_before_plan_or_snapshot_access(self):
        with patch.object(emotion, 'require_inactive', side_effect=ValueError('正在分析音乐情绪')) as guard, \
                patch.object(controller, 'read_plan') as read:
            with self.assertRaisesRegex(ValueError, '正在分析音乐情绪'):
                controller.start('unused', 'a' * 32, {}, None)
        guard.assert_called_once_with()
        read.assert_not_called()

    def test_idle_start_continues_to_existing_plan_validation(self):
        with patch.object(emotion, 'require_inactive') as guard, \
                patch.object(controller, 'read_plan', return_value={'approved': False}):
            with self.assertRaisesRegex(ValueError, '保存并确认'):
                controller.start('unused', 'b' * 32, {}, None)
        guard.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
