from app.llm.client import LLMClientFactory, LLMConfigFactory, LLMClient
import asyncio

REWRITE_PROMPT = """你是搜索查询优化器。把用户问题改写为适合检索知识库的简洁查询。
要求：
- 保留核心实体与意图，去掉客套话（"那个啥""怎么办呀"）
- 改为关键词式，2-8 个词
- 只输出改写后的查询，不要解释、不要引号

原问题：{question}
改写后："""


async def rewrite_query(llm: LLMClient, question: str) -> str:
    resp = await llm.chat(
        [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}]
    )
    return resp.content.strip()

async def main():
    llm = LLMClientFactory.create(LLMConfigFactory.from_settings())
    q = await rewrite_query(llm, '那个啥，我上次买的那个柜子想退，运费谁出啊？')
    q1= await rewrite_query(llm, '我之前买的衣柜想不要了，退回去快递费要我自己掏吗？')
    q2= await rewrite_query(llm, '柜子到货尺寸不合适打算退货，来回运费是商家承担不？')
    print('改写后:', q)
    print('改写后:', q1)
    print('改写后:', q2)


if __name__ == "__main__":
    asyncio.run(main())