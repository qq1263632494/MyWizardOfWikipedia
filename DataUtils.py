# 数据处理的相关工具
from typing import List, Dict


# 为本任务专门准备的数据结构
# 包括对方的utterance，
# 自己的回复response，
# 由32句文档组成的知识池knowledge_pool，
# 选择知识的索引choosen_index，
# 选择的知识golden_sentence
class WizardOfWikipediaExample:
    def __init__(self, utterance: str,
                 response: str,
                 knowledge_pool: List[str],
                 choosen_index: int,
                 golden_sentence: str,
                 topic: str):
        self.utterance = utterance
        self.response = response
        self.knowledge_pool = knowledge_pool
        self.choosen_index = choosen_index
        self.golden_sentence = golden_sentence
        self.topic = topic


def compile_one_dialog(ITEM: Dict) -> List[WizardOfWikipediaExample]:
    """
        将一段对话转换为若干个WizardOfWikipediaExample对象
        e.g. data = read_json(数据路径)
             examples = compile_one_dialog(data[0])
        :param ITEM: 从json读取出的数据中的某一个
        :return: 为处理后的WizardOfWikipediaExample对象的列表
        :rtype List[WizardOfWikipediaExample]
    """
    TOPIC = ITEM['chosen_topic']
    PERSONA = ITEM['persona']
    DIALOGS = ITEM['dialog']
    TOPIC_KNOWLEDGES = ITEM['chosen_topic_passage']
    LAST_UTTER = PERSONA.lower()
    datas = []
    for ind, DIALOG in enumerate(DIALOGS):
        if DIALOG['speaker'][2:] == 'Apprentice':
            LAST_UTTER = DIALOG['text']
        else:
            utterance = LAST_UTTER.lower()
            response = DIALOG['text'].lower()
            knowledge_pool = []
            knowledge_pool += TOPIC_KNOWLEDGES
            retrieved_passages = DIALOG['retrieved_passages']
            for passage in retrieved_passages:
                key = list(passage.keys())
                knowledge_pool += passage[key[0]]
            golden_sentence = DIALOG['checked_sentence']
            keys = list(golden_sentence.keys())
            if len(keys) != 0:
                golden_sentence = golden_sentence[keys[0]].lower()
            else:
                golden_sentence = 'no_passages_used'
            new_pool = []
            for sentence in knowledge_pool:
                new_pool.append(sentence.lower())
            if golden_sentence == 'no_passages_used':
                choosen_index = -1
            else:
                try:
                    choosen_index = new_pool.index(golden_sentence)
                except Exception:
                    choosen_index = -2
            datas.append(WizardOfWikipediaExample(utterance,
                                                  response,
                                                  new_pool,
                                                  choosen_index,
                                                  golden_sentence,
                                                  TOPIC))
    return datas


def display_example(example: WizardOfWikipediaExample):
    """
        显示一个WizardOfWikipediaExample的内容
        :param example: 一个WizardOfWikipediaExample对象
    """
    print('[TOPIC]')
    print(example.topic)
    print('[Utterance]')
    print(example.utterance)
    print('[Response]')
    print(example.response)
    print('[Knowledge Pool]')
    print(example.knowledge_pool)
    print('[Choosen Index]')
    print(example.choosen_index)
    print('[Golden Sentence]')
    print(example.golden_sentence)


# 将全部对话转换为若干个WizardOfWikipediaExample对象
def compile_all_dialogs(ALL_DATA: Dict) -> List[WizardOfWikipediaExample]:
    """
        将全部对话转换为对应的WizardOfWikipediaExample对象
        e.g. data = read_json(数据路径)
             examples = compile_one_dialog(data)
        :param ALL_DATA: 从json读取出的数据中
        :return: 为处理后的WizardOfWikipediaExample对象的列表
        :rtype List[WizardOfWikipediaExample]
    """
    all_datas = []
    for e in ALL_DATA:
        all_datas += compile_one_dialog(e)
    for e in all_datas:
        e.knowledge_pool = e.knowledge_pool[0:32]
        if e.choosen_index >= 32:
            e.choosen_index = -3
    return all_datas


# 将全部的WizardOfWikipediaExample对象堆叠为句子的列表
def from_all_data_to_all_sentences(all_datas: List[WizardOfWikipediaExample]) -> List[str]:
    """
        将全部对话转换为对应的WizardOfWikipediaExample对象
        :param all_datas: 全部的WizardOfWikipediaExample对象
        :return: 为处理后的全部的句子
        :rtype List[str]
    """
    all_sent = []
    for e in all_datas:
        all_sent.append(e.utterance)
        all_sent.append(e.response)
        all_sent += e.knowledge_pool
    return all_sent
