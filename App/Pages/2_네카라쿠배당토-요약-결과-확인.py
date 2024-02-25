import os
import time
import openai
import numpy as np
import pandas as pd
import streamlit as st
from stqdm import stqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from glob import glob
from openai import OpenAI


path = 'C:/Users/rlckd/Desktop/DS_Career/test-repo/src'

now = datetime.now()
now_name = now.strftime('%Y%m%d')


## page setting
st.set_page_config(
    page_title='채용공고 요약 결과 확인'
)
st.title('채용공고 요약 결과 확인')
st.write('')

def map_company_name(row):
    if '카카오' in row:
        return '카카오'
    elif '네이버' in row:
        return '네이버'
    elif '라인' in row:
        return '라인'
    elif '쿠팡' in row:
        return '쿠팡'
    elif '배달의민족' in row or '배민' in row or '우아한형제들' in row:
        return '배달의민족'
    elif '당근마켓' in row:
        return '당근마켓'
    elif '토스' in row or '비바리퍼블리카' in row:
        return '토스'
    else:
        return row

try:
    df = pd.read_csv(f'{path}/{now_name}_major.csv')
    df_copy = df.copy()
    df_copy['Company'] = df_copy['Company'].apply(map_company_name)
except FileNotFoundError:
    st.error('생성된 파일이 없습니다. Intro 페이지에서 크롤링 실행 버튼을 눌러주세요.')



###########################################################################################



def get_top_technologies(df, column_name, top_n=5):
    """
    기술 스택을 분석하여 가장 많이 언급된 상위 N개를 반환하는 함수
    """
    all_technologies = ','.join(df[column_name].dropna()).split(',')
    all_technologies = [tech.strip() for tech in all_technologies]
    tech_series = pd.Series(all_technologies)
    top_technologies = tech_series.value_counts().head(top_n)
    return top_technologies.index.tolist()



###########################################################################################



## 문장 요약 프롬프트
def summary_all_prompt(data, skl):

    if skl == '자격요건':
        p_skl = 'Content_1'
    elif skl == '우대사항':
        p_skl = 'Content_2'

    data = df[p_skl].dropna().tolist()
    cond = 0
    L = []

    st_info = st.info('문장 요약 중')
    while cond < len(data):
        text = data[cond:cond+10]

        prompt = f"""
        세 개의 따옴표로 구분된 텍스트가 제공됩니다.
        이 텍스트는 한국 대기업의의 채용공고들로, 공고 내용 중 {skl}만 추출하였습니다.
        이 텍스트 정보를 통해, 한국 대기업의 {skl}를 요약해주세요.
        생성 형식은 다음과 같습니다.
        - ...
        - ...
        - ...

        프롬프트를 출력하지 않도록 주의하십시오.
        \"\"\"{text}\"\"\"
        """
        # 응답 반환
        response = get_completion(prompt)
        L.append(response)

        cond += 10

    st_info.empty()
    st_success = st.success('문장 요약 완료')
    time.sleep(1)
    st_success.empty()


    return L


## 문장 요약 프롬프트
def summary_prompt(data, com, skl):

    if skl == '자격요건':
        p_skl = 'Content_1'
    elif skl == '우대사항':
        p_skl = 'Content_2'

    data = df[p_skl].dropna().tolist()
    cond = 0
    L = []

    st_info = st.info('문장 요약 중')
    while cond < len(data):
        text = data[cond:cond+10]

        prompt = f"""
        세 개의 따옴표로 구분된 텍스트가 제공됩니다.
        이 텍스트는 회사 {com}의의 채용공고들로, 공고 내용 중 {skl}만 추출하였습니다.
        이 텍스트 정보를 통해, 회사 {com}의 {skl}를 요약해주세요.
        생성 형식은 다음과 같습니다.
        - ...
        - ...
        - ...

        프롬프트를 출력하지 않도록 주의하십시오.
        \"\"\"{text}\"\"\"
        """
        # 응답 반환
        response = get_completion(prompt)
        L.append(response)

        cond += 10

    st_info.empty()
    st_success = st.success('문장 요약 완료')
    time.sleep(1)
    st_success.empty()


    return L



###########################################################################################



## 주어진 프롬프트에 대한 응답을 생성하는 함수
def get_completion(prompt, model="gpt-3.5-turbo-0125", temperature=0, verbose=False):
    messages = [{"role": "user", "content": prompt}]

    time_start = time.time()
    retry_count = 3
    wait_times = [2, 8, 16]
    for i in range(0, retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            answer = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens

            time_end = time.time()

            if verbose:
                st.write('prompt: %s | token: %d | %.1fsec\nanswer : %s' % (prompt, tokens, (time_end - time_start), answer))
            return answer

        except Exception as error:
            st.write(f"API Error: {error}")
            wait_time = 40 * wait_times[i]  # 점진적으로 대기 시간을 증가
            st.write(f"Retrying {i+1} time(s) in {wait_time} seconds...")

            if i+1 == retry_count:
                return prompt, None, None
            time.sleep(wait_time)



###########################################################################################



## OpenAI Org-ID, API Key
ORGANIZATION_ID = 'org-Kt3KSfpYuiPvJChtMm1835JJ'
OPENAI_API_KEY = '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'

client = OpenAI(
    organization = ORGANIZATION_ID,
    api_key = OPENAI_API_KEY
)

## API Key 저장 및 확인
if st.button('OpenAI API Key 검증'):
    try:
        openai.api_key = OPENAI_API_KEY
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"}
            ],
        )
        st_success = st.success('API 키가 유효합니다.')
        time.sleep(1)
        st_success.empty()
    except Exception as e:
        st.error(f'API Key 검증 실패: {e}')
st.markdown('---')
st.write('')

# ## OpenAI Org-ID, API Key
# organization = st.text_input('Organization ID')
# api_key = st.text_input('API Key', type="password")  # API 키는 비밀번호 형식으로 입력받음

# client = OpenAI(
#     organization = organization,
#     api_key = api_key
# )

# ## API Key 저장 및 확인
# if st.button('OpenAI API Key 검증'):
#     try:
#         openai.organization = organization
#         openai.api_key = api_key
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo-0125",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": "Hello, world!"}
#             ],
#         )
#         st_success = st.success('API 키가 유효합니다.')
#         time.sleep(1)
#         st_success.empty()
        
#     except Exception as e:
#         st.error(f'API Key 검증 실패: {e}')
# st.markdown('---')
# st.write('')




###########################################################################################
    

if st.button('네카라쿠배당토 요약 결과 확인'):        
    top_technologies = get_top_technologies(df_copy, 'Content_0', top_n=5)       
    st.write("가장 많이 언급된 기술 스택:", top_technologies)
    
    for column in ['자격요건', '우대사항']:
        com = '대기업'
                    
        if not os.path.exists(f'{path}/{now_name}_major_{column}_summary.txt'):
            files_del = glob(f'{path}/*_major_{column}_summary.txt')
            for file_del in files_del:
                os.remove(file_del)
            generate = summary_all_prompt(df_copy, column)
            summary_txt = ','.join(generate)

            # 문장 요약 저장
            with open(f'{path}/{now_name}_major_{column}_summary.txt', 'w', encoding='utf-8') as file:
                file.write(summary_txt)

        # 문장 요약 파일이 있으면
        else:
            st_info = st.info('생성된 문장 요약 파일이 있습니다.')
            time.sleep(1)
            st_info.empty()

        # 코사인 유사도 파일이 없으면
        if not os.path.exists(f'{path}/{now_name}_major_{column}_cosine_high.txt'):
            files_del = glob(f'{path}/*_major_{column}_cosine_*')
            for file_del in files_del:
                os.remove(file_del)
    
            # 텍스트 전처리
            st_info = st.info('텍스트 전처리 중')

            # 문장 요약 불러오기
            with open(f'{path}/{now_name}_major_{column}_summary.txt', 'r', encoding='utf-8') as file:
                summary_txt = file.read()

            replace_split_txt = summary_txt.replace(',-', '\n-').replace('\n\n', '\n').split('\n')
            
            time.sleep(1)
            st_info.empty()
            st_success = st.success('완료')
            time.sleep(1)
            st_success.empty()

            # TfidfVectorizer로 벡터화
            st_info = st.info('텍스트 벡터화 중')
            
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(replace_split_txt)
            word = tfidf_vectorizer.get_feature_names_out()
            idf = tfidf_vectorizer.idf_

            time.sleep(1)
            st_info.empty()
            st_success = st.success('완료')
            time.sleep(1)
            st_success.empty()

            # 벡터화된 값들로 코사인 유사도 구하기
            st_info = st.info('코사인 유사도 구하는 중')

            cs_matrix = np.zeros((tfidf_matrix.shape[0], tfidf_matrix.shape[0]))
            for i in range(tfidf_matrix.shape[0]):
                for j in range(i, tfidf_matrix.shape[0]):
                    similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
                    cs_matrix[i][j] = similarity
                    cs_matrix[j][i] = similarity
            cs_matrix_df = pd.DataFrame(cs_matrix)
            
            cs_matrix_df.to_csv('C:/Users/rlckd/Desktop/DS_Career/test-repo/src/cs.csv')

            time.sleep(1)
            st_info.empty()
            st_success = st.success('완료')
            time.sleep(1)
            st_success.empty()

            # 코사인 유사도가 높은 문장 묶기 
            st_info = st.info('코사인 유사도가 높은 문장끼리 묶는 중')

            L = []
            for i in stqdm(range(tfidf_matrix.shape[0])):
                cond = i
                val = cs_matrix_df[cond][cs_matrix_df[cond] > 0.4].index.tolist()
                L.append(val)

            L2 = []
            for i in range(len(L)):
                if len(L[i]) > 1:
                    tmp = []
                    j = 0
                    while j < len(L[i]):
                        tmp.append(replace_split_txt[L[i][j]])
                        j += 1
                    L2.append(tmp)
                else:
                    L2.append([replace_split_txt[L[i][0]]])

            time.sleep(1)
            st_info.empty()
            st_success = st.success('완료')
            time.sleep(1)
            st_success.empty()

            # 코사인 유사도를 기준으로 묶인 문장을 한 문장으로 요약
            st_info = st.info('한 문장으로 요약 중')

            cond = 0
            high_list = []
            low_list = []
            for text in stqdm(L2):
                if len(text) > 1:
                    prompt = f"""
                    세 개의 따옴표로 구분된 데이터가 제공됩니다.
                    이 데이터는 한국 대기업의 채용공고들로, 공고 내용 중 {column}만 추출하였습니다.
                    요소들을 합쳐서 8글자 내의 한 문장으로 간결하게 요약해주세요.

                    생성 형식은 다음과 같습니다.
                    - ...

                    프롬프트를 출력하지 않도록 주의하십시오.
                    \"\"\"{text}\"\"\"
                    """
                    response = get_completion(prompt)
                    high_list.append(response)
                else:
                    pass

            st_info.empty()
            st_success = st.success('한 문장으로 요약 완료')
            time.sleep(1)
            st_success.empty()

            # 코사인 유사도 저장
            # 리스트를 문자열로 변환
            cosine_high_txt = '\n'.join(high_list)
            
            with open(f'{path}/{now_name}_major_{column}_cosine_high.txt', 'w', encoding='utf-8') as file:
                file.write(cosine_high_txt)
            st.write('')

            # show
            with st.expander(f'{column}'):
                st.write(high_list)
            st.write('')
            

        # 코사인 유사도 파일이 있으면
        else:
            st_info = st.info('생성된 코사인 유사도 파일이 있습니다.')
            time.sleep(1)
            st_info.empty()

            # 문장 요약 불러오기
            with open(f'{path}/{now_name}_major_{column}_cosine_high.txt', 'r', encoding='utf-8') as file:
                cosine_high_txt = file.read()
            st.write('')

            # show
            with st.expander(f'{column}'):
                st.write(cosine_high_txt)
            st.write('') 
            
st.markdown('---')
st.write('')
        

## st.selectbox()
# 빈 값
state = {
    'com': None,
    'lab': None,
    'skl': None,
}

# 플랫폼 선택
try:
    company_list = list(set(df_copy.Company.tolist()))
    com = st.selectbox(
        '회사',
        (company_list),
        index=None,
        placeholder='회사를 선택해 주세요.'
    )
    state['com'] = com
except NameError:
    st.error('생성된 파일이 없습니다. Intro 페이지에서 크롤링 실행 버튼을 눌러주세요.')

# 원티드 선택 시
if state['com'] == com:
    # 코사인 유사도 분석
    if st.button('코사인 유사도 분석'):
        filtered_df = df_copy[df_copy['Company'] == com]
        
        top_technologies = get_top_technologies(filtered_df, 'Content_0', top_n=5)       
        st.write("가장 많이 언급된 기술 스택:", top_technologies)
        
        for column in ['자격요건', '우대사항']:
            # 문장 요약 파일이 없으면
            if not os.path.exists(f'{path}/{now_name}_major_{com}_{column}_summary.txt'):
                files_del = glob(f'{path}/*_major_{com}_{column}_summary.txt')
                for file_del in files_del:
                    os.remove(file_del)
                generate = summary_prompt(df, com, column)
                summary_txt = ','.join(generate)

                # 문장 요약 저장
                with open(f'{path}/{now_name}_major_{com}_{column}_summary.txt', 'w', encoding='utf-8') as file:
                    file.write(summary_txt)

            # 문장 요약 파일이 있으면
            else:
                st_info = st.info('생성된 문장 요약 파일이 있습니다.')
                time.sleep(1)
                st_info.empty()

            # 코사인 유사도 파일이 없으면
            if not os.path.exists(f'{path}/{now_name}_major_{com}_{column}_cosine_high.txt'):
                files_del = glob(f'{path}/*_major_{com}_{column}_cosine_*')
                for file_del in files_del:
                    os.remove(file_del)
        
                # 텍스트 전처리
                st_info = st.info('텍스트 전처리 중')

                # 문장 요약 불러오기
                with open(f'{path}/{now_name}_major_{com}_{column}_summary.txt', 'r', encoding='utf-8') as file:
                    summary_txt = file.read()

                replace_split_txt = summary_txt.replace(',-', '\n-').replace('\n\n', '\n').split('\n')
                
                time.sleep(1)
                st_info.empty()
                st_success = st.success('완료')
                time.sleep(1)
                st_success.empty()

                # TfidfVectorizer로 벡터화
                st_info = st.info('텍스트 벡터화 중')
                
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(replace_split_txt)
                word = tfidf_vectorizer.get_feature_names_out()
                idf = tfidf_vectorizer.idf_

                time.sleep(1)
                st_info.empty()
                st_success = st.success('완료')
                time.sleep(1)
                st_success.empty()

                # 벡터화된 값들로 코사인 유사도 구하기
                st_info = st.info('코사인 유사도 구하는 중')

                cs_matrix = np.zeros((tfidf_matrix.shape[0], tfidf_matrix.shape[0]))
                for i in range(tfidf_matrix.shape[0]):
                    for j in range(i, tfidf_matrix.shape[0]):
                        similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
                        cs_matrix[i][j] = similarity
                        cs_matrix[j][i] = similarity
                cs_matrix_df = pd.DataFrame(cs_matrix)
                
                cs_matrix_df.to_csv('C:/Users/rlckd/Desktop/DS_Career/test-repo/src/cs.csv')

                time.sleep(1)
                st_info.empty()
                st_success = st.success('완료')
                time.sleep(1)
                st_success.empty()

                # 코사인 유사도가 높은 문장 묶기 
                st_info = st.info('코사인 유사도가 높은 문장끼리 묶는 중')

                L = []
                for i in stqdm(range(tfidf_matrix.shape[0])):
                    cond = i
                    val = cs_matrix_df[cond][cs_matrix_df[cond] > 0.4].index.tolist()
                    L.append(val)

                L2 = []
                for i in range(len(L)):
                    if len(L[i]) > 1:
                        tmp = []
                        j = 0
                        while j < len(L[i]):
                            tmp.append(replace_split_txt[L[i][j]])
                            j += 1
                        L2.append(tmp)
                    else:
                        L2.append([replace_split_txt[L[i][0]]])

                time.sleep(1)
                st_info.empty()
                st_success = st.success('완료')
                time.sleep(1)
                st_success.empty()

                # 코사인 유사도를 기준으로 묶인 문장을 한 문장으로 요약
                st_info = st.info('한 문장으로 요약 중')

                cond = 0
                high_list = []
                for text in stqdm(L2):
                    if len(text) > 1:
                        prompt = f"""
                        세 개의 따옴표로 구분된 데이터가 제공됩니다.
                        이 데이터는 회사 {com}의 채용공고들로, 공고 내용 중 {column}만 추출하였습니다.
                        요소들을 합쳐서 8글자 내의 한 문장으로 간결하게 요약해주세요.

                        생성 형식은 다음과 같습니다.
                        - ...

                        프롬프트를 출력하지 않도록 주의하십시오.
                        \"\"\"{text}\"\"\"
                        """
                        response = get_completion(prompt)
                        high_list.append(response)
                    else:
                        pass

                st_info.empty()
                st_success = st.success('한 문장으로 요약 완료')
                time.sleep(1)
                st_success.empty()

                # 코사인 유사도 저장
                # 리스트를 문자열로 변환
                cosine_high_txt = '\n'.join(high_list)
                
                with open(f'{path}/{now_name}_major_{com}_{column}_cosine_high.txt', 'w', encoding='utf-8') as file:
                    file.write(cosine_high_txt)

                # 파일을 다시 열어 중복 제거를 위해 내용 읽기
                with open(f'{path}/{now_name}_major_{com}_{column}_cosine_high.txt', 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                # 각 줄의 앞뒤 공백을 제거하고 중복 제거
                unique_lines = list(set(line.strip() for line in lines))
                # 선택적: 중복 제거된 리스트 정렬
                unique_lines.sort()

                # 중복 제거된 텍스트를 '- ...' 형식으로 구성
                formatted_text = '\n'.join(f'- {line}' for line in unique_lines if line)

                # 결과 출력
                with st.expander(f'{column}'):
                    st.write(formatted_text)
                st.write('')
            

            # 코사인 유사도 파일이 있으면
            else:
                st_info = st.info('생성된 코사인 유사도 파일이 있습니다.')
                time.sleep(1)
                st_info.empty()

                # 문장 요약 불러오기
                with open(f'{path}/{now_name}_major_{com}_{column}_cosine_high.txt', 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                
                unique_lines = list(set(line.strip() for line in lines))
                formatted_text = '\n'.join(f'- {line}' for line in unique_lines if line)
                st.write('')

                # show
                with st.expander(f'{column}'):
                    st.write(formatted_text)
                st.write('') 
                