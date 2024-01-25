import streamlit as st
import plotly.express as px
import openai
import pandas as pd
from inference import gpt_query_stream
from streamlit_pills import pills
import json
from utils.general_utils import set_png_as_page_bg, display_props

# set_png_as_page_bg('static/background.png')

display_props()
# API for Dr Liao
# client = openai.AzureOpenAI(
#     api_key="10df9e66aea744219b6eab9074bc56fa",
#     api_version="2023-12-01-preview",
#     azure_endpoint="https://gc-openai-je.openai.azure.com/"
# )
# openai_model = "gpt4-for-DrLiao"

# API for YumC
client = openai.AzureOpenAI(
    api_key="3ddf85f581e142709dd8129aef144679",
    api_version="2023-07-01-preview",
    azure_endpoint="https://gpt4-for-shaojie.openai.azure.com/"
)
openai_model = "gpt4-shaojie"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = openai_model

lang = ("CN", 'EN')
options = list(range(len(lang)))
st.session_state['language'] = st.sidebar.radio("语言/Language", options, format_func=lambda x: lang[x])

st.sidebar.info(f'''Logged in: **Edgar Cao**\ncao.edgar@bcg.com''')
st.sidebar.divider()


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def load_css(file_name: str):
    with open(file_name) as file:
        st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

# 使用你的CSS文件名替换下面的'style.css'
# load_css('./style/style.css')

if "file_path" not in st.session_state:
    st.session_state.file_path = './data/extracted_牛蛙塔可_v5.xlsx'
    st.session_state.selected_product = "牛蛙塔可"
    st.session_state.summary_file_path = './data/牛蛙塔可_summary.json'


# if "prod_display_name" in st.session_state:
#     st.sidebar.write("选择产品:" + st.session_state.prod_display_name)

def set_product_frog():
    st.session_state.selected_product = "牛蛙塔可"
    if st.session_state['language'] == 0:
        st.session_state.prod_display_name = "牛蛙塔可"
        st.session_state.summary_file_path = './data/牛蛙塔可_summary.json'
    if st.session_state['language'] == 1:
        st.session_state.prod_display_name = "Bullfrog Taco"
        st.session_state.summary_file_path = './data/牛蛙塔可_summary_en.json'
    st.session_state.file_path = './data/extracted_牛蛙塔可_v5.xlsx'


def set_product_crawfish():
    st.session_state.selected_product = "小龙虾塔可"
    if st.session_state['language'] == 0:
        st.session_state.prod_display_name = "小龙虾塔可"
        st.session_state.summary_file_path = './data/小龙虾塔可_summary.json'
    if st.session_state['language'] == 1:
        st.session_state.prod_display_name = "Crawfish Taco"
        st.session_state.summary_file_path = './data/小龙虾塔可_summary_en.json'
    st.session_state.file_path = './data/extracted_小龙虾塔可_v5.xlsx'


def set_product_k_sa():
    st.session_state.selected_product = "大盘鸡K萨"
    if st.session_state['language'] == 0:
        st.session_state.prod_display_name = "大盘鸡K萨"
        st.session_state.summary_file_path = './data/大盘鸡K萨_summary.json'
    if st.session_state['language'] == 1:
        st.session_state.prod_display_name = "Braised Chicken K Pizza"
        st.session_state.summary_file_path = './data/大盘鸡K萨_summary_en.json'
    st.session_state.file_path = './data/extracted_大盘鸡K萨_v5.xlsx'


# main page
if st.session_state['language'] == 0:

    mapper = {
        'product': {
            'Bullfrog Taco': '牛蛙塔可',
            'Crawfish Taco': '小龙虾塔可',
            'Braised Chicken K Pizza': '大盘鸡K萨'
        }
    }

    if 'prod_display_name' not in st.session_state:
        st.session_state['prod_display_name'] = None
    elif st.session_state['prod_display_name'] in mapper['product']:
        st.session_state['prod_display_name'] = mapper['product'][st.session_state['prod_display_name']]

    st.sidebar.header("消费者洞察分析")
    st.sidebar.button("**牛蛙塔可** \n\n",
                      use_container_width=True,
                      on_click=set_product_frog)
    st.sidebar.button("**小龙虾塔可** \n\n",
                      use_container_width=True,
                      on_click=set_product_crawfish)
    st.sidebar.button("**大盘鸡K萨** \n\n ",
                      use_container_width=True,
                      on_click=set_product_k_sa)

    product_name = st.session_state.selected_product
    display_name = st.session_state.prod_display_name
    summary = load_json(st.session_state.summary_file_path)

    if display_name in ['牛蛙塔可', '小龙虾塔可', '大盘鸡K萨']:

        st.markdown(f'''## :green[{product_name} - 消费者洞察分析]''')
        st.text('')
        st.text('')

        df = pd.read_excel(st.session_state.file_path, sheet_name='Sheet1', engine='openpyxl')
        df_cleaned = df.dropna()
        df_cleaned = df_cleaned[df_cleaned['entity'] == product_name]
        df_cleaned = df_cleaned[df_cleaned['category'] != '综合']

        st.write('请选择洞察维度：')
        op1, op2= st.columns(2)
        # with op1:
        #     views = ['All', '产品视角', '非产品视角']
        #     sel_view = st.selectbox('洞察视角', options=['All', '产品视角', '非产品视角'])
        #     if sel_view == '产品视角':
        #         df_cleaned = df_cleaned[df_cleaned['product_related'] == 1]
        #     elif sel_view == '非产品视角':
        #         df_cleaned = df_cleaned[df_cleaned['product_related'] == 0]
        with op1:
            categories = ['All'] + df_cleaned['category'].drop_duplicates().tolist()
            sel_category = st.selectbox('一级标签', categories)
            if sel_category != 'All':
                df_cleaned = df_cleaned[df_cleaned['category'] == sel_category]
        with op2:
            parts = ['All'] + df_cleaned['parts'].drop_duplicates().tolist()
            sel_part = st.selectbox('组成部分', parts)
            if sel_part != 'All':
                df_cleaned = df_cleaned[df_cleaned['parts'] == sel_part]

        # "op", option
        if sel_category == 'All':
            st.session_state.reset_summary = False
        else:
            st.session_state.reset_summary = False

        # 负向评论
        pos_reviews = df_cleaned[df_cleaned['sentiment'] == '正向']
        neg_reviews = df_cleaned[df_cleaned['sentiment'] == '负向']
        neg_cat_aggregation = neg_reviews.groupby('category').size().reset_index(name='count')
        neg_cat_aggregation = neg_cat_aggregation.sort_values(by='count', ascending=False)
        pos_cat_aggregation = pos_reviews.groupby('category').size().reset_index(name='count')
        pos_cat_aggregation = pos_cat_aggregation.sort_values(by='count', ascending=False)

        print('='*40)
        print(neg_cat_aggregation)
        print('-'*40)
        print(pos_cat_aggregation)
        print('='*40)

        st.text('')
        st.text('')
        pos_percent = f"{100*pos_reviews.comment.nunique()/df_cleaned.comment.nunique():.2f}%"
        neg_percent = f"{100*neg_reviews.comment.nunique()/df_cleaned.comment.nunique():.2f}%"

        # Tag Summary Section
        st.markdown(f'''#### :green[1. 一级标签总结报告]''')
        with st.expander('**点击展开总结报告**'):
            if sel_category == 'All':
                text = summary['All']
                st.markdown(f'##### 综合观点', unsafe_allow_html=True)
            else:
                text = summary[sel_category]
                st.markdown(f'##### {sel_category}维度观点', unsafe_allow_html=True)

            html_text = text.replace('\n', '<p/>')
            st.markdown(f'##### {html_text}', unsafe_allow_html=True)

        # 如果category选择All，需要展开一级维度
        st.markdown(f'''#### :green[2. 洞察维度分布]''')
        if sel_category == 'All':
            with st.expander("**点击展开洞察维度 - 一级标签**"):
                neg = st.checkbox("**负向观点**: 提及 " + str(len(neg_reviews)) + "次，占比" + neg_percent, value=True, key='1')
                pos = st.checkbox("**正向观点**: 提及 " + str(len(pos_reviews)) + "次，占比" + pos_percent, value=True, key='2')
                if pos and not neg:
                    df_filtered = pos_reviews
                if neg and not pos:
                    df_filtered = neg_reviews
                if pos and neg:
                    df_filtered = df_cleaned
                if neg:
                    chart_data_neg = pd.DataFrame(
                        {
                            "维度": neg_cat_aggregation['category'].tolist(),
                            "提及次数": neg_cat_aggregation['count'].tolist()
                        }
                    )
                    chart_data_neg = chart_data_neg.sort_values(by='提及次数', ascending=True)
                    fig = px.bar(chart_data_neg, y='维度', x='提及次数', orientation='h', title=f'负向评论维度分布 (共{chart_data_neg.shape[0]}个标签)', color_discrete_sequence=["#f6737c"])
                    # fig = px.pie(chart_data, values='提及次数', names='维度', title='正向标签统计')
                    st.plotly_chart(fig)

                if pos:
                    chart_data_pos = pd.DataFrame(
                        {
                            "维度": pos_cat_aggregation['category'].tolist(),
                            "提及次数": pos_cat_aggregation['count'].tolist()
                        }
                    )
                    chart_data_pos = chart_data_pos.sort_values(by='提及次数', ascending=True)
                    fig = px.bar(chart_data_pos, y='维度', x='提及次数', orientation='h', title=f'正向评论维度分布 (共{chart_data_pos.shape[0]}个标签)', color_discrete_sequence=["#09A5AD"])
                    # fig = px.pie(chart_data, values='提及次数', names='维度', title='正向标签统计')
                    st.plotly_chart(fig)

        # 展开二级维度
        with st.expander("**点击展开标签分布 - 二级标签**"):
            neg = st.checkbox("**负向观点**: 提及 " + str(len(neg_reviews)) + "次，占比" + neg_percent, value=True, key='3')
            pos = st.checkbox("**正向观点**: 提及 " + str(len(pos_reviews)) + "次，占比" + pos_percent, value=True, key='4')
            if pos and not neg:
                df_filtered = pos_reviews
            if neg and not pos:
                df_filtered = neg_reviews
            if pos and neg:
                df_filtered = df_cleaned
            if neg:
                neg_tag_aggregation = neg_reviews.groupby('tag').size().reset_index(name='count')
                neg_tag_aggregation = neg_tag_aggregation.sort_values(by='count', ascending=False)
                chart_data_neg = pd.DataFrame(
                    {
                        "标签": neg_tag_aggregation['tag'].tolist(),
                        "提及次数": neg_tag_aggregation['count'].tolist()
                    }
                ).sort_values(by='提及次数', ascending=True)
                fig = px.bar(chart_data_neg, y='标签', x='提及次数', orientation='h', title=f'负向标签统计 (共{chart_data_neg.shape[0]}个标签)', color_discrete_sequence=["#f6737c"], width=700)
                st.plotly_chart(fig)
            if pos:
                pos_tag_aggregation = pos_reviews.groupby('tag').size().reset_index(name='count')
                chart_data_pos = pd.DataFrame(
                    {
                        "标签": pos_tag_aggregation['tag'].tolist(),
                        "提及次数": pos_tag_aggregation['count'].tolist()
                    }
                ).sort_values(by="提及次数", ascending=True)
                fig = px.bar(chart_data_pos, y='标签', x='提及次数', orientation='h', title=f'正向标签统计 (共{chart_data_pos.shape[0]}个标签)', color_discrete_sequence=["#09A5AD"], width=700)
                st.plotly_chart(fig)

        # aggregated_counts = df_filtered.groupby(['tag', 'sentiment']).size().unstack(fill_value=0).reset_index()
        # result_array = aggregated_counts.to_numpy()
        # result_list = result_array.tolist()
        #
        # result_list_modified = [inner[1:] for inner in result_list]
        # x_axis_keys = [inner[0] for inner in result_list]
        # result_array_modified = np.array(result_list_modified)


        # pill section
        st.markdown(f'''#### :green[3. 二级标签下钻洞察]''')
        neg = st.checkbox("**负向观点**: 提及 " + str(len(neg_reviews)) + "次，占比" + neg_percent, value=True, key='5')
        pos = st.checkbox("**正向观点**: 提及 " + str(len(pos_reviews)) + "次，占比" + pos_percent, value=True, key='6')
        if pos and not neg:
            df_filtered = pos_reviews
        if neg and not pos:
            df_filtered = neg_reviews
        if pos and neg:
            df_filtered = df_cleaned
        if not pos and not neg:
            df_filtered = df_cleaned
        pills_ops = df_filtered['tag'].dropna().value_counts().reset_index()
        pills_ops.columns = ['tag', 'count']

        pills_ops_with_count = pills_ops.sort_values(by='count', ascending=False)
        pills_to_display = [f"{pill_count['tag']}: {pill_count['count']}" for _, pill_count in pills_ops_with_count.iterrows()]

        emojis = ["🍀", "🎈", "🌈"]
        emoji_list = []

        for _, row in pills_ops_with_count.iterrows():
            if row['count'] >= 10:
                emoji_list.append(emojis[1])
            elif row['count'] > 5:
                emoji_list.append(emojis[0])
            else:
                emoji_list.append(emojis[2])

        sel_pill = pills("", options=pills_to_display, icons=emoji_list, clearable=True)
        if sel_pill:
            sel_tag = sel_pill.split(":")[0].replace(" ", "")
            df_table = df_filtered.loc[df_filtered['tag'] == sel_tag]

        st.markdown(f"**标签为 :green[{sel_tag}] 的原始评论：共{len(df_table)}条**")
        df_table = df_table[['chunk', 'category', 'parts', 'sentiment', 'tag', 'comment']]
        df_table.columns = ['Chunk', 'Category', 'Parts', 'Sentiment', 'Tag', 'Comment']
        df_table.reset_index(drop=True, inplace=True)
        reviews = '\n\n'.join(df_table['Chunk'].tolist())

        if st.button('生成总结'):
            st.session_state.reset_summary = True
            system_prompt = f'''
        你是一位Consumer Insights分析专家，你的客户是一家炸鸡快餐店，最近他们推出了一款新产品:{product_name}，
        他们想要了解消费者对于他们的产品在{sel_tag}这个观点的具体看法。
            '''
            summarize_prompt = f'''
        你的任务是:
        - 根据用户评论，总结观点，告诉我用户观点主要集中在哪些方面，哪些方面占比比较高
        - 总结需要结构化,以bullet形式输出，要求逻辑严谨，观点不重复不遗漏，MECE
        - don't make up an answer，观点要有输入用户评论作为依据
        
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

            messages = [{
                "role": 'system',
                "content": system_prompt
            }, {
                "role": 'user',
                "content": summarize_prompt
            }]

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

            for item in gpt_query_stream(client, messages):
                response_text += item
                message_col.write(response_text)

        with st.expander('点击展开消费者原始评论'):
            st.markdown("<div style='height:20px'> </div>", unsafe_allow_html=True)
            df_table.index = df_table.index + 1
            df_table = df_table[['Comment','Chunk','Sentiment','Parts','Category','Tag']]
            df_table.columns = ['原始评论', '语块切分', '情感分类', '组成部分', '一级标签', '二级标签']
            st.table(df_table)

if st.session_state['language'] == 1:

    mapper = {'product': {
        '牛蛙塔可': 'Bullfrog Taco',
        '小龙虾塔可': 'Crawfish Taco',
        '大盘鸡K萨': 'Braised Chicken K Pizza'
    }, 'category': {
        '综合': 'General',
        '服务': 'Service',
        '口感': 'Texture',
        '味道': 'Taste',
        '份量': 'Amount',
        '外观': 'Appearance',
        '食材': 'Materials',
        '食品安全': 'Safety',
        '价格': 'Price',
        'All': 'All'
    }, 'component_Bullfrog Taco': {
        '牛蛙塔可整体': 'Bullfrog Taco',
        '鸡排': 'Chicken Steak',
        '牛蛙': 'Bullfrog',
        '酱料': 'Dressing',
        '生菜': 'Lettuce',
        '辣椒粉': 'Chili Powder',
        'All': 'All'
    }, 'component_Crawfish Taco': {
        '小龙虾塔可整体': 'Crawfish Taco',
        '小龙虾': 'Crawfish',
        '鸡排': 'Chicken Steak',
        '酸笋': 'Pickled Bamboo Shoots',
        '酱料': 'Dressing',
        '生菜': 'Lettuce',
        'All': 'All'
    }, 'component_Braised Chicken K Pizza': {
        '大盘鸡K萨整体': 'Braised Chicken K Pizza',
        '面饼': 'Cracker',
        '大盘鸡酱料': 'Dressing',
        '芝士': 'Cheese',
        '鸡米花': 'Popcorn Chicken',
        '土豆泥': 'Mashed Potatoes',
        'All': 'All'
    }, 'tags': json.loads(open('./data/tag_translation.json', 'r').read())}

    if 'prod_display_name' not in st.session_state:
        st.session_state['prod_display_name'] = None
    elif st.session_state['prod_display_name'] in mapper['product']:
        st.session_state['prod_display_name'] = mapper['product'][st.session_state['prod_display_name']]

    st.sidebar.header("LLM VoC Insights Analysis")
    st.sidebar.button("**Bullfrog Taco** \n\n",
                      use_container_width=True,
                      on_click=set_product_frog)
    st.sidebar.button("**Crawfish Taco** \n\n",
                      use_container_width=True,
                      on_click=set_product_crawfish)
    st.sidebar.button("**Braised Chicken K Pizza** \n\n ",
                      use_container_width=True,
                      on_click=set_product_k_sa)

    product_name = st.session_state.selected_product
    display_name = st.session_state.prod_display_name
    summary = load_json(st.session_state.summary_file_path)

    if display_name in ['Bullfrog Taco', 'Crawfish Taco', 'Braised Chicken K Pizza']:

        st.markdown(f'''## :green[{display_name}]''')
        st.text('')
        st.text('')

        df = pd.read_excel(st.session_state.file_path, sheet_name='Sheet1', engine='openpyxl')
        df['tag'] = df['tag'].map(mapper['tags'])
        df_cleaned = df.dropna()
        df_cleaned = df_cleaned[df_cleaned['entity'] == product_name]
        df_cleaned = df_cleaned[df_cleaned['category'] != '综合']

        st.write('Please select dimension：')
        op1, op2 = st.columns(2)
        # with op1:
        #     views = ['All', 'Product Related', 'Non Product Related']
        #     sel_view = st.selectbox('Product or Not', options=['All', 'Product Related', 'Non Product Related'])
        #     if sel_view == 'Product Related':
        #         df_cleaned = df_cleaned[df_cleaned['product_related'] == 1]
        #     elif sel_view == 'Non Product Related':
        #         df_cleaned = df_cleaned[df_cleaned['product_related'] == 0]
        with op1:
            cat_mapper = mapper['category']
            cat_reverse_mapper = {cat_mapper[x]:x for x in cat_mapper}
            categories = ['All'] + [cat_mapper[x] for x in df_cleaned['category'].drop_duplicates().tolist()]
            sel_category = cat_reverse_mapper[st.selectbox('Tag Level 1', categories)]
            if sel_category != 'All':
                df_cleaned = df_cleaned[df_cleaned['category'] == sel_category]
        with op2:
            parts_mapper = mapper[f'component_{display_name}']
            parts_reverse_mapper = {parts_mapper[x]: x for x in parts_mapper}
            parts = ['All'] +[parts_mapper[x] for x in df_cleaned['parts'].drop_duplicates().tolist()]
            sel_part = parts_reverse_mapper[st.selectbox('Components', parts)]
            if sel_part != 'All':
                df_cleaned = df_cleaned[df_cleaned['parts'] == sel_part]

        # "op", option
        if sel_category == 'All':
            st.session_state.reset_summary = False
        else:
            st.session_state.reset_summary = False

        # 负向评论
        pos_reviews = df_cleaned[df_cleaned['sentiment'] == '正向']
        neg_reviews = df_cleaned[df_cleaned['sentiment'] == '负向']
        neg_cat_aggregation = neg_reviews.groupby('category').size().reset_index(name='count')
        neg_cat_aggregation = neg_cat_aggregation.sort_values(by='count', ascending=False)
        pos_cat_aggregation = pos_reviews.groupby('category').size().reset_index(name='count')
        pos_cat_aggregation = pos_cat_aggregation.sort_values(by='count', ascending=False)

        st.text('')
        st.text('')
        pos_percent = f"{100 * pos_reviews.comment.nunique() / df_cleaned.comment.nunique():.2f}%"
        neg_percent = f"{100 * neg_reviews.comment.nunique() / df_cleaned.comment.nunique():.2f}%"

        # Tag Summary Section
        st.markdown(f'''#### :green[1. Tag Level 1 Reports]''')
        with st.expander('**Click to Expand - General CI Reports**'):
            if sel_category == 'All':
                text = summary['All']
                st.markdown(f'##### General Opinions', unsafe_allow_html=True)
            else:
                text = summary[sel_category]
                st.markdown(f'##### Opinions on {cat_mapper[sel_category]}', unsafe_allow_html=True)

            html_text = text.replace('\n', '<p/>')
            st.markdown(f'##### {html_text}', unsafe_allow_html=True)

        # 如果category选择All，需要展开一级维度
        st.markdown(f'''#### :green[2. Tags Distribution]''')
        if sel_category == 'All':
            # st.text(summary['general']['text'])
            with st.expander("**Click to Expand - Distribution on Tag Level 1**"):
                neg = st.checkbox("**Negative opinions**: " + str(len(neg_reviews)) + "times mentioned, " + neg_percent,
                                  value=True, key='1')
                pos = st.checkbox("**Positive opinions**: " + str(len(pos_reviews)) + "times mentioned, " + pos_percent,
                                  value=True, key='2')
                if pos and not neg:
                    df_filtered = pos_reviews
                if neg and not pos:
                    df_filtered = neg_reviews
                if pos and neg:
                    df_filtered = df_cleaned
                if neg:
                    chart_data_neg = pd.DataFrame(
                        {
                            "维度": neg_cat_aggregation['category'].tolist(),
                            "提及次数": neg_cat_aggregation['count'].tolist()
                        }
                    )
                    chart_data_neg = chart_data_neg.sort_values(by='提及次数', ascending=True)
                    chart_data_neg.columns = ['Category', 'Counts']
                    chart_data_neg['Category'] = chart_data_neg['Category'].map(mapper['category'])
                    fig = px.bar(chart_data_neg, y='Category', x='Counts', orientation='h', title=f'Negative Comments ({chart_data_neg.shape[0]} tags)',
                                 color_discrete_sequence=["#f6737c"])
                    # fig = px.pie(chart_data, values='提及次数', names='维度', title='正向标签统计')
                    st.plotly_chart(fig)

                if pos:
                    chart_data_pos = pd.DataFrame(
                        {
                            "维度": pos_cat_aggregation['category'].tolist(),
                            "提及次数": pos_cat_aggregation['count'].tolist()
                        }
                    )
                    chart_data_pos = chart_data_pos.sort_values(by='提及次数', ascending=True)
                    chart_data_pos.columns = ['Category', 'Counts']
                    chart_data_pos['Category'] = chart_data_pos['Category'].map(mapper['category'])
                    fig = px.bar(chart_data_pos, y='Category', x='Counts', orientation='h', title=f'Positive Comments ({chart_data_pos.shape[0]} tags)',
                                 color_discrete_sequence=["#09A5AD"])
                    # fig = px.pie(chart_data, values='提及次数', names='维度', title='正向标签统计')
                    st.plotly_chart(fig)

        # 展开二级维度
        with st.expander("**Click to Expand - Distribution on Tag Level 2**"):
            neg = st.checkbox("**Negative opinions**: " + str(len(neg_reviews)) + "times mentioned, " + neg_percent, value=True, key='3')
            pos = st.checkbox("**Positive opinions**: " + str(len(pos_reviews)) + "times mentioned, " + pos_percent, value=True, key='4')
            if pos and not neg:
                df_filtered = pos_reviews
            if neg and not pos:
                df_filtered = neg_reviews
            if pos and neg:
                df_filtered = df_cleaned
            if neg:
                neg_tag_aggregation = neg_reviews.groupby('tag').size().reset_index(name='count')
                neg_tag_aggregation = neg_tag_aggregation.sort_values(by='count', ascending=False)
                chart_data_neg = pd.DataFrame(
                    {
                        "标签": neg_tag_aggregation['tag'].tolist(),
                        "提及次数": neg_tag_aggregation['count'].tolist()
                    }
                ).sort_values(by='提及次数', ascending=True)
                chart_data_neg.columns = ['Tags', 'Counts']
                fig = px.bar(chart_data_neg, y='Tags', x='Counts', orientation='h', title=f'Negative Comments ({chart_data_neg.shape[0]} tags)',
                             color_discrete_sequence=["#f6737c"], width=700)
                st.plotly_chart(fig)
            if pos:
                pos_tag_aggregation = pos_reviews.groupby('tag').size().reset_index(name='count')
                chart_data_pos = pd.DataFrame(
                    {
                        "标签": pos_tag_aggregation['tag'].tolist(),
                        "提及次数": pos_tag_aggregation['count'].tolist()
                    }
                ).sort_values(by="提及次数", ascending=True)
                chart_data_pos.columns = ['Tags', 'Counts']
                fig = px.bar(chart_data_pos, y='Tags', x='Counts', orientation='h', title=f'Positive Comments ({chart_data_pos.shape[0]} tags)',
                             color_discrete_sequence=["#09A5AD"], width=700)
                st.plotly_chart(fig)

        # aggregated_counts = df_filtered.groupby(['tag', 'sentiment']).size().unstack(fill_value=0).reset_index()
        # result_array = aggregated_counts.to_numpy()
        # result_list = result_array.tolist()
        #
        # result_list_modified = [inner[1:] for inner in result_list]
        # x_axis_keys = [inner[0] for inner in result_list]
        # result_array_modified = np.array(result_list_modified)


        # pill section
        st.markdown(f'''#### :green[3. Tag Level 2 Deep Dive]''')
        neg = st.checkbox("**Negative opinions**: " + str(len(neg_reviews)) + "times mentioned, " + neg_percent, value=True,
                          key='5')
        pos = st.checkbox("**Positive opinions**: " + str(len(pos_reviews)) + "times mentioned, " + pos_percent, value=True,
                          key='6')
        if pos and not neg:
            df_filtered = pos_reviews
        if neg and not pos:
            df_filtered = neg_reviews
        if pos and neg:
            df_filtered = df_cleaned
        if not pos and not neg:
            df_filtered = df_cleaned
        pills_ops = df_filtered['tag'].dropna().value_counts().reset_index()
        pills_ops.columns = ['tag', 'count']

        pills_ops_with_count = pills_ops.sort_values(by='count', ascending=False)
        pills_to_display = [f"{pill_count['tag']}: {pill_count['count']}" for _, pill_count in pills_ops_with_count.iterrows()]

        emojis = ["🍀", "🎈", "🌈"]
        emoji_list = []

        for _, row in pills_ops_with_count.iterrows():
            if row['count'] >= 10:
                emoji_list.append(emojis[1])
            elif row['count'] > 5:
                emoji_list.append(emojis[0])
            else:
                emoji_list.append(emojis[2])

        sel_pill = pills("", options=pills_to_display, icons=emoji_list, clearable=True)
        if sel_pill:
            sel_tag = sel_pill.split(":")[0]
            df_table = df_filtered.loc[df_filtered['tag'] == sel_tag]

        st.markdown(f"**Comments tagged as :green[{sel_tag}]: {len(df_table)} in total**")
        df_table = df_table[['chunk', 'category', 'parts', 'sentiment', 'tag', 'comment']]
        df_table.columns = ['Chunk', 'Category', 'Parts', 'Sentiment', 'Tag', 'Comment']
        df_table.reset_index(drop=True, inplace=True)
        reviews = '\n\n'.join(df_table['Chunk'].tolist())

        if st.button('Summarize'):
            st.session_state.reset_summary = True
            system_prompt = f'''
            你是一位Consumer Insights分析专家，你的客户是一家炸鸡快餐店，最近他们推出了一款新产品:{product_name}，
            他们想要了解消费者对于他们的产品在{sel_tag}这个观点的具体看法。
                '''
            summarize_prompt = f'''
            你的任务是:
            - 根据用户评论，总结观点，告诉我用户观点主要集中在哪些方面，哪些方面占比比较高
            - 总结需要结构化,以bullet形式输出，要求逻辑严谨，观点不重复不遗漏，MECE
            - don't make up an answer，观点要有输入用户评论作为依据
            - please answer in English
        
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
            
            - please answer in English
                '''

            messages = [{
                "role": 'system',
                "content": system_prompt
            }, {
                "role": 'user',
                "content": summarize_prompt
            }]

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

            for item in gpt_query_stream(client, messages):
                response_text += item
                message_col.write(response_text)

        with st.expander('Expand to see original customer comments'):
            st.markdown("<div style='height:20px'> </div>", unsafe_allow_html=True)
            df_table['Category'] = df_table['Category'].map(mapper['category'])
            df_table['Parts'] = df_table['Parts'].map(mapper[f'component_{display_name}'])
            df_table['Sentiment'] = df_table['Sentiment'].map({'正向':'Positive','负向':'Negative'})
            df_table.index = df_table.index + 1
            df_table = df_table[['Comment', 'Chunk', 'Sentiment', 'Parts', 'Category', 'Tag']]
            df_table.columns = ['Raw Comment', 'Chunk', 'Sentiment', 'Components', 'Tag Level 1', 'Tag Level 2']
            st.table(df_table)
