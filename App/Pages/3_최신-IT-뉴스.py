import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

today = datetime.now().date()

def parse_date(date_str):
    if '일 전' in date_str:
        days_ago = int(date_str.replace('일 전', ''))
        date = today - timedelta(days=days_ago)
    elif '시' in date_str and '분' in date_str:
        date = today
    else:
        date = None
    return date
    
def is_within_7_days(date):
    return today - date <= timedelta(days=7)

async def parse(html):
    soup = BeautifulSoup(html, 'html.parser')
    news_data = []
    news_items = soup.select('div.node-list > div')
    for item in news_items:
        date_str = item.select_one('p.card-text > small').text.strip()
        parsed_date = parse_date(date_str)
        
        if parsed_date and is_within_7_days(parsed_date):
            title_element = item.select_one('div.col > div > h5 > a')
            tag_element = item.select_one('div.col > div > p.card-text.d-flex.align-items-center.justify-content-between > span > a:nth-child(1)')
            
            if title_element and tag_element:
                title = title_element.text.strip()
                tag = tag_element.text.strip()
                article_url = title_element['href']
                
                display_date = parsed_date.strftime('%Y-%m-%d')
                
                news_data.append({'title': title, 'tag': tag, 'url': article_url, 'date': display_date})
    return news_data

async def crawl_website(session, base_url):
    page = 1
    all_news_data = []
    while True:
        url = f"{base_url}?page={page}"
        html = await fetch(session, url)
        news_data = await parse(html)
        if not news_data:
            break  # No more data to process
        all_news_data.extend(news_data)
        page += 1
    return all_news_data

async def main(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [crawl_website(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Streamlit 앱 설정
path = '../src'  # 현재 작업 디렉토리에 맞게 경로 설정
now = datetime.now()
now_name = now.strftime('%Y%m%d')

st.set_page_config(page_title='최신 IT 뉴스 모음')
st.title('최신 IT 뉴스 모음')

st.markdown('<br><br>', unsafe_allow_html=True)

urls = [
    'https://www.itworld.co.kr/t/54649/%EB%8D%B0%EC%9D%B4%ED%84%B0%E3%86%8D%EB%B6%84%EC%84%9D',
    'https://www.itworld.co.kr/t/69500/AI%E3%86%8DML'
]

# 비동기 작업 실행
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
news_data = loop.run_until_complete(main(urls))

news_df = pd.DataFrame(sum(news_data, []))  # 데이터 병합
news_df['date'] = pd.to_datetime(news_df['date'])  # 날짜 칼럼을 datetime 객체로 변환
news_df.sort_values(by='date', ascending=False, inplace=True)  # 내림차순으로 정렬
news_df['date'] = news_df['date'].dt.strftime('%Y-%m-%d')

csv_path = os.path.join(path, f'{now_name}_news.csv')
news_df.to_csv(csv_path, index=False)  # 데이터 저장

# 스트림릿 페이지에 데이터 표시
for index, row in news_df.iterrows():
    # 제목을 클릭하면 해당 URL로 이동하는 링크 생성
    link = f"[{row['title']}](https://www.itworld.co.kr/{row['url']})"
    st.markdown(link, unsafe_allow_html=True)

    # 태그와 날짜 정보를 표시하고, 간격 조정을 위한 HTML과 CSS 적용
    tag_date_html = f"""
    <p style='margin-top:5px;margin-bottom:5px;'>태그: {row['tag']} | 날짜: {row['date']}</p>
    """
    st.markdown(tag_date_html, unsafe_allow_html=True)
    
    # 구분선 추가
    st.markdown("<hr style='margin-top:5px;margin-bottom:10px;'/ >", unsafe_allow_html=True)
