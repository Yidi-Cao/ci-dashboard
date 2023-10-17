import streamlit as st
import openai
import pandas as pd
from streamlit_pills import pills
import numpy as np
import plotly.express as px

import matplotlib.pyplot as plt
from inference import init_gpt_4, gpt_query_stream
from summarize import summarize
import datetime
import time
import json


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


st.sidebar.subheader("Consumer Insights Analysis")
# st.sidebar.subheader("选择产品: " + st.session_state.selected_product)
if "file_path" not in st.session_state:
    st.session_state.file_path = './data/frog_taco_1.1.xlsx'
    st.session_state.selected_product = "牛蛙Taco"
    st.session_state.summary_file_path = './data/summarization_牛蛙塔可_v5.json'

if "selected_product" in st.session_state:
    st.sidebar.subheader("选择产品: " + st.session_state.selected_product)


def set_product_frog():
    st.session_state.selected_product = "牛蛙Taco"
    st.session_state.file_path = './data/extracted_牛蛙塔可_v5.xlsx'
    st.session_state.summary_file_path = './data/summarization_牛蛙塔可_v5.json'


def set_product_crawfish():
    st.session_state.selected_product = "小龙虾Taco"
    st.session_state.file_path = './data/extracted_小龙虾塔可_v5.xlsx'
    st.session_state.summary_file_path = './data/summarization_小龙虾塔可_v5.json'


def set_product_k_sa():
    st.session_state.selected_product = "K萨"
    st.session_state.file_path = './data/extracted_大盘鸡K萨_v5.xlsx'
    st.session_state.summary_file_path = './data/summarization_大盘鸡K萨_v5.json'


st.sidebar.button("**牛蛙taco**",
                  use_container_width=True,
                  on_click=set_product_frog)
st.sidebar.button("**小龙虾taco** \n\n",
                  use_container_width=True,
                  on_click=set_product_crawfish)
st.sidebar.button("**k萨** \n\n ",
                  use_container_width=True,
                  on_click=set_product_k_sa)

st.sidebar.info(f'''Logged in: **Shaojie Li**''')
st.text(" ")
st.text(" ")
st.text(" ")
# st.sidebar.caption(f"Selected client: **{st.session_state.client_name}**")

# selected_date = st.sidebar.date_input(
#     "Selected date",
#     # value=dt.today(),
#     value=datetime.date(2023, 4, 4),
#     label_visibility='visible')

# 设置你的OpenAI的API密钥
openai.api_key = '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'

product_name = '大盘鸡K萨'


summary = load_json(st.session_state.summary_file_path)

# 从Excel文件中加载Sheet1
df = pd.read_excel(st.session_state.file_path,
                   sheet_name='Sheet1', engine='openpyxl')
product_related = st.selectbox(
    '产品相关',
    ['all', '产品', '非产品']
)
# df_cleaned = (df[(df['category'] != '无标签')]).dropna()
df_cleaned = df
if product_related == '产品':
    df_cleaned = df_cleaned[df_cleaned['product_related'] == 1]
elif product_related == '非产品':
    df_cleaned = df_cleaned[df_cleaned['product_related'] == 0]

options_candidate = df_cleaned['category'].drop_duplicates().tolist()
options_candidate = ['all'] + options_candidate

parts_candidate = df_cleaned['parts'].drop_duplicates().tolist()
parts_candidate = ['all'] + parts_candidate

op1, op2 = st.columns(2)
with op1:
    option = st.selectbox(
        '维度',
        options_candidate
    )

with op2:
    parts = st.selectbox('组成部分', parts_candidate)

df_filtered = df_cleaned
# "op", option
if option == 'all':
    df_filtered = df_cleaned
else:
    df_filtered = df_filtered[df_filtered['category'] == option]

# "parts", parts
if parts == 'all':
    df_filtered = df_filtered
else:
    df_filtered = df_filtered[df_filtered['parts'] == parts]

pos_reviews = df_filtered[df_filtered['sentiment'] == '正向']
neg_reviews = df_filtered[df_filtered['sentiment'] == '负向']


neg_cat_aggregation = neg_reviews.groupby(
    'category').size().reset_index(name='count')
neg_cat_aggregation = neg_cat_aggregation.sort_values(
    by='count', ascending=False)

# ========================================================
if option == 'all':
    chart_data = pd.DataFrame(
        {
            "维度": neg_cat_aggregation['category'].tolist(),
            "提及次数": neg_cat_aggregation['count'].tolist()
        }
    )
    # st.bar_chart(chart_data, y="提及次数", x="标签")
    chart_data = chart_data.sort_values(by='提及次数', ascending=True)
    fig = px.bar(chart_data, y='维度', x='提及次数', orientation='h',
                 title='负向维度统计', color_discrete_sequence=["#f6737c"])

    # fig = px.pie(chart_data, values='提及次数', names='维度', title='正向标签统计')
    st.plotly_chart(fig)

    pos_cat_aggregation = pos_reviews.groupby(
        'category').size().reset_index(name='count')
    pos_cat_aggregation = pos_cat_aggregation.sort_values(
        by='count', ascending=False)

    chart_data = pd.DataFrame(
        {
            "维度": pos_cat_aggregation['category'].tolist(),
            "提及次数": pos_cat_aggregation['count'].tolist()
        }
    )
    chart_data = chart_data.sort_values(by='提及次数', ascending=True)
    fig = px.bar(chart_data, y='维度', x='提及次数', orientation='h',
                 title='正向标签统计', color_discrete_sequence=["#09A5AD"])

    # fig = px.pie(chart_data, values='提及次数', names='维度', title='正向标签统计')
    st.plotly_chart(fig)


pos_percent = f"{100*len(pos_reviews)/len(df_filtered):.2f}%"
neg_percent = f"{100*len(neg_reviews)/len(df_filtered):.2f}%"

"总评论数", df_filtered['comment'].nunique()

option, "主要观点"
if option == 'all':
    st.text(summary['general'])
else:
    st.text(summary['byCategory'][option])


neg = st.checkbox("负向提及 " + str(len(neg_reviews)) +
                  "次 " + neg_percent, value=True)
pos = st.checkbox("正向提及 " + str(len(pos_reviews)) +
                  "次 " + pos_percent, value=True)

pos_neg_count = 0

if pos and neg:
    pos_neg_count = 2
elif pos:
    pos_neg_count = 1
    df_filtered = pos_reviews
elif neg:
    pos_neg_count = 1
    df_filtered = neg_reviews


tag_count_map = {}
for item in df_filtered['tag'].dropna():
    if item not in tag_count_map:
        tag_count_map[item] = 1
    else:
        tag_count_map[item] += 1

aggregated_counts = df_filtered.groupby(
    ['tag', 'sentiment']).size().unstack(fill_value=0)
aggregated_counts_reset = aggregated_counts.reset_index()
result_array = aggregated_counts_reset.to_numpy()
result_list = result_array.tolist()

result_list_modified = [inner[1:] for inner in result_list]
x_axis_keys = [inner[0] for inner in result_list]
result_array_modified = np.array(result_list_modified)


# chart section
if neg:
    neg_tag_aggregation = neg_reviews.groupby(
        'tag').size().reset_index(name='count')
    neg_tag_aggregation = neg_tag_aggregation.sort_values(
        by='count', ascending=False)

    # st.markdown("<span style='color:red'>负向评价</span>",
    #             unsafe_allow_html=True)
    chart_data = pd.DataFrame(
        {
            "标签": neg_tag_aggregation['tag'].tolist(),
            "提及次数": neg_tag_aggregation['count'].tolist()
        }
    )
    # st.bar_chart(chart_data, y="提及次数",  x="标签", color="#fc033d")
    chart_data = chart_data.sort_values(by='提及次数', ascending=True)
    fig = px.bar(chart_data, y='标签', x='提及次数', orientation='h',
                 title='负向标签统计', color_discrete_sequence=["#f6737c"])

    # 显示图表
    st.plotly_chart(fig)

if pos:
    pos_tag_aggregation = pos_reviews.groupby(
        'tag').size().reset_index(name='count')

    # st.markdown("<span style='color:navy'>正向评价</span>",
    #             unsafe_allow_html=True)
    chart_data = pd.DataFrame(
        {
            "标签": pos_tag_aggregation['tag'].tolist(),
            "提及次数": pos_tag_aggregation['count'].tolist()
        }
    ).sort_values(by="提及次数", ascending=True).reset_index(drop=True)

    # st.bar_chart(chart_data, y="提及次数", x="标签")
    chart_data = chart_data.sort_values(by='提及次数', ascending=True)
    fig = px.bar(chart_data, y='标签', x='提及次数', orientation='h',
                 title='正向标签统计', color_discrete_sequence=["#09A5AD"])

    # 显示图表
    st.plotly_chart(fig)


pills_options = df_filtered['tag'].dropna().drop_duplicates().tolist()
pills_options_with_count = [
    str(option) + " : " + str(tag_count_map[option]) for option in pills_options]

emojis = ["🍀", "🎈", "🌈"]
emoji_list = []
for op in pills_options:
    if tag_count_map[op] > 30:
        emoji_list.append(emojis[1])
    elif tag_count_map[op] > 10:
        emoji_list.append(emojis[0])
    else:
        emoji_list.append(emojis[2])
selected = pills("", options=pills_options_with_count,
                 icons=emoji_list, clearable=True)

df_table = df_filtered
if (selected):
    selected_tag = selected.split(":")[0].replace(" ", "")
    df_table = df_filtered.loc[df_filtered['tag'] == selected_tag]

'标签相关评论提及数', len(df_table)

# Tag Summary Section
reviews = df_table['chunk'].tolist()

if st.button('生成总结'):
    system_prompt_template = '''
        你是一位consumer insights分析专家，你的客户是KFC，最近他们推出了一款新产品，{product_name}
        他们想要了解消费者对于他们的产品的正向负向看法，尤其是负向看法。
    '''

    summarize_prompt_template = '''
        你的任务是:
        - 根据用户评论，总结观点，告诉我用户观点主要集中在哪些方面，哪些方面比重比较高
        - 总结需要结构化,以bullet形式输出，尽量做到MECE
        
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
    openai, engine = init_gpt_4()

    prompt = {
        "system": system_prompt_template.format(product_name=product_name),
        "user": summarize_prompt_template.format(reviews=reviews),
    }

    text_width = 800
    response_text = ""
    batch_size = 50
    batch_counter = 0
    message_col = st.empty()

    width = 400
    height = 200

    spinner_html = f'''
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://media.tenor.com/ysXbGu-PSTcAAAAC/adorable-cat.gif" alt="loading..." width="300"/>
    </div>
    '''
    message_col.markdown(
        spinner_html,
        unsafe_allow_html=True
    )

    for item in gpt_query_stream(openai, prompt, engine):
        response_text += item
        message_col.write(response_text)


# if 'product_related' in df.columns:
#     df_table = df_table.drop('product_related')

# df_table = df_table.drop(
#     columns=['token_size', 'entity', 'product_related', 'inference_time', 'opinion'])


df_table = df_table[['chunk', 'category',
                     'parts', 'sentiment', 'tag', 'comment']]
df_table.reset_index(drop=True, inplace=True)
st.markdown("<div style='height:20px'> </div>", unsafe_allow_html=True)
st.table(df_table)

# chart_data = pd.DataFrame(
#     {
#         "col1": list(range(20)) * 3,
#         "col2": np.random.randn(60),
#         "col3": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
#     }
# )

# st.bar_chart(chart_data, x="col1", y="col2", color="col3")

# col1, col2 = st.columns(2)
# with col1:
#     chart_data = pd.DataFrame(
#         50*np.abs(np.random.randn(20, 3)),
#         columns=['棉花', '玉米', '大豆'])
#     if option == 'all':
#         st.line_chart(chart_data)
#     else:
#         st.line_chart(chart_data[option])

# with col2:
#     chart_data = pd.DataFrame(
#         50 * np.abs(np.random.randn(20, 3)),
#         columns=['棉花', '玉米', '大豆'])
#     if option == 'all':
#         st.bar_chart(chart_data)
#     else:
#         st.bar_chart(chart_data[option])


# if len(pos_reviews) > 0 and pos_neg_count == 2:
#     chart_data = pd.DataFrame(
#         {
#             "提及次数": result_array_modified.flatten(),
#             "标签": np.repeat(x_axis_keys, 2),
#             "情感倾向": np.tile(['正', '负'], len((tag_count_map.keys())))
#         }
#     )
#     st.bar_chart(chart_data, x="标签", y="提及次数", color="情感倾向")
# else:
#     chart_data = pd.DataFrame(
#         {
#             "提及次数": tag_count_map.values(),
#             "标签": tag_count_map.keys()
#         }
#     )
#     st.bar_chart(chart_data, x="标签", y="提及次数")
