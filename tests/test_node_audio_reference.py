import copy
import unittest
from unittest.mock import patch

from test_plugin import nodes, sample_plan


class NodeAudioReferenceTests(unittest.TestCase):
    def run_node(self, visual_type='performance', slot=1, source='loader'):
        plan = sample_plan()
        plan['approved'] = True
        material = {'visual_type': visual_type}
        outputs = (None, None, 121, 'fixture', material, *((None,) * 6), '', 24., *((None,) * 3))
        graph = {
            'loader': {'class_type': 'H3LVUnified', 'inputs': {}},
            'reference': {'class_type': 'MiniMaxH3ReferenceToVideo',
                          'inputs': {'ref_audios.ref_audio_0': [source, slot]}},
        }
        with patch.object(nodes, 'read_plan', return_value=copy.deepcopy(plan)), \
                patch.object(nodes, 'data_root', return_value='fixture-projects'), \
                patch.object(nodes.LoadSegment, 'load', return_value=outputs):
            result = nodes.Unified().process({}, 'singing', 15, 12, '', '',
                                             project_id=plan['id'], prompt=graph, unique_id='loader')
        self.assertEqual(len(result), len(nodes.Unified.RETURN_TYPES))
        return result[4]

    def test_hidden_graph_metadata_does_not_add_visible_ports(self):
        schema = nodes.Unified.INPUT_TYPES()
        self.assertEqual(schema['hidden'], {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'})
        self.assertNotIn('prompt', schema['required'])

    def test_actual_slot_one_connection_confirms_audio(self):
        material = self.run_node()
        self.assertIs(material['audio_reference_available'], True)
        self.assertEqual(material['audio_reference_slot'], 1)

    def test_wrong_connection_or_nonperformer_cannot_claim_audio(self):
        for changes in ({'slot': 0}, {'slot': True}, {'source': 'other'}, {'visual_type': 'atmosphere'}):
            with self.subTest(**changes):
                material = self.run_node(**changes)
                self.assertIs(material['audio_reference_available'], False)
                self.assertIsNone(material['audio_reference_slot'])

    def test_downstream_audio_rewire_invalidates_cached_material(self):
        plan = sample_plan()
        plan['approved'] = True
        graph = {'loader': {'class_type': 'H3LVUnified', 'inputs': {}},
                 'reference': {'class_type': 'MiniMaxH3ReferenceToVideo',
                               'inputs': {'ref_audios.ref_audio_0': ['loader', 1]}}}
        with patch.object(nodes, 'read_plan', return_value=plan), \
                patch.object(nodes, 'data_root', return_value='fixture-projects'):
            before = nodes.Unified.IS_CHANGED(plan['id'], 0, prompt=graph, unique_id='loader')
            graph['reference']['inputs']['ref_audios.ref_audio_0'] = ['other', 0]
            after = nodes.Unified.IS_CHANGED(plan['id'], 0, prompt=graph, unique_id='loader')
        self.assertNotEqual(before, after)


if __name__ == '__main__':
    unittest.main()
