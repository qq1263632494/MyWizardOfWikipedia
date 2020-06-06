# 处理文件的相关工具
import json


def read_json(PATH):
    """
        读取json文件的工具
        :param PATH: json文件的路径
        :return: 反序列化得到的字典
        :rtype Dict
        """
    with open(PATH) as f:
        return json.load(f)
