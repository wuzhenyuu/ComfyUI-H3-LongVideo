"""Current generation and historical cache fixtures exercise separate contracts."""
import test_materials as fixtures

REMOTE_TEXT = ('subject_definitions:\n<Subject 1> from <Picture 1>. <Audio 1> is singing.\n'
               'summary:\nOne shot follows <Audio 1>.\nretention_analysis:\nKeep scene.\n'
               'detailed_description:\n[Shot 1] Move.\noverall_soundscape:\nN/A\nnon_diegetic_music:\nN/A')


def chinese_prompt(count=2, performance=False):
    sources = '、'.join(f'<Picture {i+1}>' for i in range(count))
    text = fixtures.TEXT.replace('来自 <Picture 1>', '来自 '+sources)
    if performance:
        text = text.replace('\n\nsummary:', '\n<Audio 1> 是 <Subject 1> (S1) 已核实的独立项目演唱人声参考。\n\nsummary:')
        text = text.replace('[reference generation]', '[reference generation + audio reference]')
        text = text.replace('\n\ndetailed_description:', '\n<Audio 1>：reference - 参考已核实的项目人声节奏。\n\ndetailed_description:')
        text = text.replace(fixtures.SHOT_BODY, '<Subject 1> (S1) 根据 <Audio 1> 进行自然演唱表演，'+fixtures.SHOT_BODY)
    return text


def confirmed_packet(*args, **kwargs):
    return {**fixtures.materials.packet(*args, **kwargs),
            'audio_reference_available': True, 'audio_reference_slot': 1}
