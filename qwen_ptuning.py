import json
import os
import time
from abc import ABC
from enum import Enum

import requests
from loguru import logger


# Create an enum for different P-Tuning task categories.
class LoraType(Enum):
    Nothing = 0
    Classify = 1
    Keywords = 2
    NL2SQL = 3


MODEL_NAMES_MAPPING = {
    LoraType.Classify: "intent",
    LoraType.Keywords: "keywords",
    LoraType.NL2SQL: "nl2sql",
    LoraType.Nothing: "Qwen1.5_7B",
}

MAX_PROMPT_LEN = 5120
DEFAULT_QWEN_API_URL = "http://localhost:8089/v1/completions"


# QwenLoRA class: one base model with multiple LoRA adapters.
class QwenLoRA(ABC):

    # Select different adapters by task type.
    def __init__(self, ptuning_type: LoraType):
        super().__init__()
        self.ptuning_type = ptuning_type
        self.model_name = MODEL_NAMES_MAPPING[ptuning_type]
        self.api_url = os.getenv("QWEN_API_URL", DEFAULT_QWEN_API_URL)
        self.timeout = int(os.getenv("QWEN_API_TIMEOUT", "30"))

    def request(self, prompt):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": f"Human: {prompt}\nAssistant: ",
            "stop": "<|endoftext|>",
            "temperature": 0,
            "max_tokens": 64,
            "top_k": 1,
        }

        start = time.time()
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.error("Qwen API request failed: {}", e)
            return "Error"

        elapsed = time.time() - start
        logger.debug("Qwen API latency: {:.3f}s", elapsed)

        choices = result.get("choices", [])
        if not choices:
            logger.error("Qwen API response has no choices: {}", result)
            return "Error"
        return choices[0].get("text", "").strip()

    def _llm_type(self) -> str:
        return "Qwen"

    def __call__(self, prompt: str) -> str:
        if len(prompt) > MAX_PROMPT_LEN:
            prompt = prompt[:MAX_PROMPT_LEN]
        return self.request(prompt)

    def _get_classify_prompt(self, question) -> str:
        classify_prompt = '''
        请问“{}”是属于下面哪个类别的问题?
        A: 公司基本信息,包含股票简称, 公司名称, 外文名称, 法定代表人, 注册地址, 办公地址, 公司网址网站, 电子信箱等.
        B: 公司员工信息,包含员工人数, 员工专业, 员工类别, 员工教育程度等.
        C: 财务报表相关内容, 包含资产负债表, 现金流量表, 利润表 中存在的字段, 包括费用, 资产，金额，收入等.
        D: 计算题,无法从年报中直接获得,需要根据计算公式获得, 包括增长率, 率, 比率, 比重,占比等.
        E: 统计题，需要从题目获取检索条件，在数据集/数据库中进行检索、过滤、排序后获得结果.
        F: 开放性问题,包括介绍情况,介绍方法,分析情况,分析影响,什么是XXX.
        你只需要回答字母编号, 不要回答字母编号及选项文本外的其他内容.
        '''.format(question)
        return classify_prompt

    # Load classification fine-tuned weights to return a question category label.
    def classify(self, question: str):
        classify_prompt = self._get_classify_prompt(question)
        response = self.request(classify_prompt)
        return response

    def _get_keywords_prompt(self, question) -> str:
        question_prompt = '''
                请帮我从以下句子中提取关键词。这些关键词是句子中最重要、最能概括句子主题的词汇。通过这些关键词，你可以更好地理解句子的内容。你只需要回答文本中的关键词,不要回答其他内容.
                用户输入：
                '''
        keywords_prompt = f"{question_prompt} {question}"
        return keywords_prompt

    # Load keyword-extraction fine-tuned weights to return extracted keywords.
    # For lookup/calculation questions return core terms; for statistics return DB-retrieval fields; for open questions return regular keywords.
    def keywords(self, question: str):
        keywords_prompt = self._get_keywords_prompt(question)
        response = self.request(keywords_prompt)
        return response

    @property
    def _get_nl2sql_prompt(self) -> str:
        nl2sql_prompt = '''你是一名Mysql数据库开发人员，你精通Mysql数据库的sql代码编写，你需要根据已知的表名、字段名和用户输入的问题编写sql代码
已知表名：company_table
已知字段名：[公司全称、年份、经营活动现金流入小计、公司的中文简称、固定资产、应交税费、应付职工薪酬、未分配利润、负债合计、电子信箱、资产总计、无形资产、货币资金、资本公积、利息收入、营业收入、营业外支出、盈余公积、营业利润、营业外收入、所得税费用、其他收益、现金及现金等价物净增加额、净利润、其他应收款、营业成本、综合收益总额、流动资产合计、应收账款、预付款项、其他应付款、非流动资产合计、基本每股收益、购买商品、接受劳务支付的现金、应付账款、流动负债合计、利润总额、管理费用、其他流动资产、递延所得税资产、财务费用、营业总收入、非流动负债合计、存货、分配股利、利润或偿付利息支付的现金、稀释每股收益、所有者权益合计、营业总成本、销售费用、负债和所有者权益总计、持续经营净利润、信用减值损失、财务人员、销售人员、投资收益、行政人员、技术人员、利息费用、生产人员、研发费用、资产减值损失、递延收益、其他非流动资产、短期借款、在职员工的数量合计]
注意对问题中的中文数字（xx亿、xx千万、xx万）进行阿拉伯数字转换，如：一个亿、一亿需转换为100000000，一千万需转换为10000000，两千万需转换为20000000，一百万需转换为1000000，五千万需转换为50000000
要求sql代码中的字段名必须是已知字段名，不得新增字段名

示例模板：
"""
用户输入：2019年哪家公司的负债合计最高？

sql如下：
```sql
select 公司全称
from company_table
order by 负债合计 desc
limit 1
```
"""
请根据以下用户输入，输出sql代码。
用户输入：'''
        return nl2sql_prompt

    # Load NL2SQL fine-tuned weights to improve text-to-SQL conversion.
    def nl2sql(self, question: str):
        question_prompt = f'{self._get_nl2sql_prompt}"{question}"'
        response = self.request(question_prompt)
        if "```sql" in response:
            response = response[response.find("```sql") + 6:]
        response = response.replace("```", "").strip()
        return response


if __name__ == "__main__":
    model = QwenLoRA(LoraType.Classify)
    print(model.classify("2019年哪家公司的负债合计最高？"))

    model = QwenLoRA(LoraType.Keywords)
    print(model.keywords("2019年哪家公司的负债合计最高？"))

    model = QwenLoRA(LoraType.NL2SQL)
    print(model.nl2sql("2019年哪家公司的负债合计最高？"))

    model = QwenLoRA(LoraType.Nothing)
    print(model("你好啊！"))
