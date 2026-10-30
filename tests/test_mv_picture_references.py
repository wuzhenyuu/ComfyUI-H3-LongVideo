"""Replay the reported two-picture response without a paid API call."""
import unittest
from pathlib import Path
from unittest.mock import patch

import test_materials as fixtures
from test_expansion_fixtures import confirmed_packet, REMOTE_TEXT, chinese_prompt
materials, expansion, TEXT = fixtures.materials, fixtures.expansion, fixtures.TEXT


REPORTED = '''subject_definitions:
<Subject 1> is the young woman with a gold hairpin and a cream Hanfu. <Audio 1> is the supplied vocal reference.
summary:
One continuous performance of <Subject 1> singing to <Audio 1> in a wooden pavilion.
retention_analysis:
<Subject 1>: identity, hair and cream Hanfu are preserved from <Picture 1>.
<Picture 2>: the pavilion, tea set, lotus pond and opening composition are preserved.
detailed_description:
[Shot 1] <Subject 1> sings naturally in the pavilion shown in <Picture 2>.
overall_soundscape:
N/A
non_diegetic_music:
N/A'''


class PictureReferenceTests(unittest.TestCase):
    setUp = fixtures.MaterialTests.setUp

    def packet(self):
        return confirmed_packet(self.plan, self.plan['segments'][0], self.directory)

    def test_reported_sources_in_retention_are_accepted_with_one_subject(self):
        packet = self.packet()
        with patch.object(expansion, 'public_settings', return_value={'base_url':'https://test.invalid/v1'}):
            key = expansion.cache_key(packet, 'vision', 'model', '', 0)
            cache = Path(packet['cache_dir']); cache.mkdir(parents=True)
            (cache/(key+'.invalid.txt')).write_text(REPORTED, encoding='utf-8')
            with patch.object(expansion, 'call', side_effect=AssertionError('No paid format repair')) as request:
                result = expansion.expand(packet, 'vision', 'model', '')
        self.assertIn('<Picture 1>', result)
        self.assertIn('<Picture 2>', result)
        self.assertNotIn('<Subject 2>', result)
        self.assertIn('<Picture 1>', result.split('summary:')[0])
        self.assertIn('<Picture 2>', result.split('summary:')[0])
        request.assert_not_called()
        self.assertEqual(expansion.validate_result(result, packet), result)

    def test_new_chinese_request_keeps_both_picture_attachments(self):
        with patch.object(expansion, 'public_settings', return_value={'base_url':'https://test.invalid/v1'}), \
             patch.object(expansion, 'call', return_value={'choices':[{'message':{'content':chinese_prompt(performance=True)}}]}) as request:
            result = expansion.expand(self.packet(), 'vision', 'model', '')
        content = request.call_args.args[1]['messages'][1]['content']
        self.assertEqual([item['type'] for item in content], ['text', 'image_url', 'image_url'])
        self.assertIn('"number": 2', content[0]['text'])
        self.assertEqual(expansion.validate_result(result, self.packet()), result)

    def test_truly_missing_second_picture_is_rejected_by_number(self):
        with self.assertRaisesRegex(ValueError, 'Picture 2'):
            expansion.validate_result(TEXT, self.packet())

    def test_two_views_of_one_person_keep_a_single_subject(self):
        prompt = REPORTED.replace(
            '<Subject 1> is the young woman',
            '<Subject 1> is the same woman in <Picture 1> and <Picture 2>, the young woman')
        self.assertEqual(expansion.validate_result(prompt, self.packet()), prompt)

    def test_missing_all_sources_still_rejects_ambiguous_multi_image_output(self):
        missing = REPORTED.replace('<Picture 1>', 'the first photo').replace('<Picture 2>', 'the second photo')
        with self.assertRaisesRegex(ValueError, '多图'):
            expansion.validate_result(missing, self.packet())

    def test_out_of_range_picture_remains_an_error(self):
        with self.assertRaisesRegex(ValueError, '不存在的图片'):
            expansion.validate_result(REPORTED.replace('<Picture 2>', '<Picture 3>'), self.packet())

    def test_unbound_reference_only_in_action_does_not_invent_a_source_role(self):
        prompt = REMOTE_TEXT.replace('[Shot 1] Move.', '[Shot 1] Move beside <Picture 2>.')
        with self.assertRaisesRegex(ValueError, 'Picture 2'):
            expansion.validate_result(prompt, self.packet())


if __name__ == '__main__':
    unittest.main()
