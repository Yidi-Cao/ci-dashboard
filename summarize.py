from inference import init_gpt_4, gpt_query


system_prompt_template = '''
    你是一位consumer insights分析专家，你的客户是一家餐饮公司，
    他们想要了解消费者对于他们的产品的正向负向看法，尤其是负向看法。
'''

summarize_prompt_template = '''
    你的任务是:
    - 根据用户评论，总结观点，告诉我用户正面和负面观点主要集中在哪些方面，哪些方面比重比较高
    - 总结尽量结构化,以下是一个负面评论总结的例子，请学习例子的颗粒度和结构
    用户评论:
    {reviews}

    %例子开始
    根据消费者评论分析极氪刹车存在较多负面反馈 。主要痛点集中在 刹车
    前段过软 无法提供足够制动力 导致踩刹车的制动距离过长、 刹车反应
    不灵敏 不能够对司机操作做出即时响应 。此外 刹车噪音较大 使用过程
    中存在明显震动和抖动也影响舒适性 。 一定程度上削弱了消费者的驾驶
    信心与体验感 。这些问题反映出极氪刹车系统 在硬件设计、软件调教上
    还有优化空间 需要继续改进与调整 以提供更安全、舒适的刹车体验 满
    足消费者对动力更直接、制动更有信心的需求。
    %例子结束
'''


def summarize(reviews):
    openai, engine = init_gpt_4()

    prompt = {
        "system": system_prompt_template,
        "user": summarize_prompt_template.format(reviews=reviews),
    }
    response, time_used = gpt_query(openai, prompt, engine)
    return response, time_used
