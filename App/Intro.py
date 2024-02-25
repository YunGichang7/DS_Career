import os
import re
import time
import pandas as pd
import streamlit as st
from tqdm import tqdm
from glob import glob
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument('--disable-popup-blocking')


## page setting
st.set_page_config(
    page_title='Intro'
)
st.title('Intro')
st.write('')


## 스트림릿 사용 방법 링크 제공
st.markdown('### Streamlit 사용 방법')
st.write('이 페이지를 활용하는 방법에 대해 알아보려면 [해당 링크](https://balanced-park-5a8.notion.site/c2f2b1a1677a464b8d80e225d296f803#0a35451ece3a4dc9b9fbf4bdfd92b7b7)를 참조하세요.')
st.write('')
st.write('')


## OpenAI API Key 발급 방법 링크 제공
st.markdown('### OpenAI API Key 발급 방법')
st.write('OpenAI API Key 발급 방법에 대해 알아보려면 [해당 링크](https://quartz-beluga-805.notion.site/API-Oragnization-ID-cd8f03ffd7864dd6bfc4d1dc7b80b0b9?pvs=4)를 참조하세요.')
st.write('')
st.write('')



###########################################################################################



## 스크롤 끝까지
def scroll(driver):
    scroll_location = driver.execute_script('return document.body.scrollHeight')
    while True:
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
        time.sleep(2)
        scroll_height = driver.execute_script('return document.body.scrollHeight')
        if scroll_location == scroll_height:
            break
        else:
            scroll_location = driver.execute_script('return document.body.scrollHeight')
    driver.implicitly_wait(3)


## 스크롤 한 번만
def scroll_one(driver):
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(1)
    driver.implicitly_wait(3)


## 수집명, 수집 개수, 첫번째 값 반환
def elem_return_0(Name, List):
    return print(f'\n{Name} *** {len(List)}\n{List[0]}')


## 수집명, 수집 개수 반환
def elem_return_1(Name, List):
    return print(f'\n{Name} *** {len(List)}')



###########################################################################################



## 원티드 수집
def Wanted(KEYWORD):
    
    # webdriver 실행
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(f'https://www.wanted.co.kr/search?query={KEYWORD}&tab=position')
    driver.implicitly_wait(2)
    scroll(driver)

    # 검색 키워드 출력
    print(f'\n\n##### {KEYWORD} #####')

    # 데이터 수집
    Lin = [] # 링크
    Tit = [] # 타이틀
    Com = [] # 회사

    # 공고 개수
    num = driver.find_element(By.XPATH, '//*[@id="search_tabpanel_position"]/div/div[1]/h2').text
    num = int(num.replace('포지션', ''))
    for i in tqdm(range(1, num+1), desc='링크, 타이틀, 회사'):
        try:
            # 링크
            p_0 = driver.find_element(By.XPATH, f'//*[@id="search_tabpanel_position"]/div/div[4]/div[{i}]/a').get_attribute('href')
            Lin.append(p_0)                       

            # 타이틀
            p_1 = driver.find_element(By.XPATH, f'//*[@id="search_tabpanel_position"]/div/div[4]/div[{i}]/a/div[2]/strong').text
            Tit.append(p_1)

            # 회사
            p_2 = driver.find_element(By.XPATH, f'//*[@id="search_tabpanel_position"]/div/div[4]/div[{i}]/a/div[2]/span[1]/span').text
            Com.append(p_2)                                        
        except Exception as e:
            st.write(e)
            break

    # 데이터 출력
    elem_return_0('링크', Lin)
    elem_return_0('타이틀', Tit)
    elem_return_0('회사', Com)

    # 데이터 수집
    Loc = [] # 위치
    Ctn_0 = [] # 주요업무
    Ctn_1 = [] # 자격요건
    Ctn_2 = [] # 우대사항

    for i in tqdm(range(num), desc='위치, 주요업무, 자격요건, 우대사항'):
        driver.get(Lin[i])
        scroll_one(driver)
        time.sleep(1)
        execute = driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[1]/div/section/section/article[1]/div/button/span[2]')
        driver.execute_script("arguments[0].click();", execute)
        time.sleep(1)

        try:
            # 위치
            p_3 = driver.find_element(By.XPATH, f'//*[@id="__next"]/main/div[1]/div/section/header/div/div[1]/span[2]').text
            Loc.append(p_3)   

            # 주요업무
            p_4 = driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[1]/div/section/section/article[1]/div/div[1]').text
            Ctn_0.append(p_4)

            # 자격요건
            p_5 = driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[1]/div/section/section/article[1]/div/div[2]').text
            Ctn_1.append(p_5)

            # 우대사항
            p_6 = driver.find_element(By.XPATH, '//*[@id="__next"]/main/div[1]/div/section/section/article[1]/div/div[3]').text
            Ctn_2.append(p_6)

        except Exception as e:
            st.write(e)
            break

    driver.close()

    # 데이터 출력
    elem_return_0('위치', Loc)
    elem_return_1('주요업무', Ctn_0)
    elem_return_1('자격요건', Ctn_1)
    elem_return_1('우대사항', Ctn_2)

    # 데이터프레임 생성
    df = pd.DataFrame({
        'Title' : Tit,
        'Company' : Com,
        'Content_0' : Ctn_0, 
        'Content_1' : Ctn_1, 
        'Content_2' : Ctn_2,   
        'Link' : Lin,
        'Location' : Loc,
        'label' : f'{KEYWORD}',
        'platform': 'wanted',         
    })
    keyword_csv_file = f'{path}/{KEYWORD}.csv'
    df.to_csv(keyword_csv_file, index=False, encoding='utf-8-sig')

    return Tit, Com, Ctn_0, Ctn_1, Ctn_2, Lin, Loc



###########################################################################################



## 캐치 겉에 수집
def collect_job_information():
    # Chrome WebDriver 설정
    service = ChromeService(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service)
    # 웹 페이지 접속
    driver.get("https://www.catch.co.kr/NCS/RecruitInformation")

    # 필요한 버튼 클릭을 위한 과정
    steps = [
        ('//*[@id="Contents"]/div[3]/div/div[1]/button[4]', 1),
        ('//*[@id="Contents"]/div[3]/div/div[2]/button[3]', 1),
        ('//*[@id="Contents"]/div[3]/div/div[2]/div[3]/button', 1),
        ('//*[@id="Contents"]/div[3]/div/div[2]/div[4]/div[1]/dl/dd[3]/div[7]/a', 1),
        ('//*[@id="Contents"]/div[3]/div/div[2]/div[4]/div[1]/dl/dd[3]/div[7]/div/span[13]', 1),
        ('//*[@id="Contents"]/div[3]/div/div[2]/div[4]/div[2]/button[2]', 1)
    ]
    
    for xpath, delay in steps:
        driver.find_element(By.XPATH, xpath).click()
        time.sleep(delay)

    # 데이터를 저장할 리스트 초기화
    Com = []
    Tit = []
    Lin = []
    Cat = []

    # 페이지에 있는 모든 tr 요소를 가져옵니다.
    rows = driver.find_elements(By.XPATH, '//*[@id="recr_result"]/table/tbody/tr')
    
    # 각 행에 대해 필요한 정보 추출
    for i in range(1, len(rows) + 1):
        p_2 = driver.find_element(By.XPATH, f'//*[@id="recr_result"]/table/tbody/tr[{i}]/td[1]/p/a').text
        p_1 = driver.find_element(By.XPATH, f'//*[@id="recr_result"]/table/tbody/tr[{i}]/td[2]/a/p[1]').text
        p_0 = driver.find_element(By.XPATH, f'//*[@id="recr_result"]/table/tbody/tr[{i}]/td[2]/a').get_attribute('href')
        
        # 분류 데이터 추출
        _p_3 = driver.find_elements(By.XPATH, f'//*[@id="recr_result"]/table/tbody/tr[{i}]/td[1]/a/p[2] | //*[@id="recr_result"]/table/tbody/tr[{i}]/td[2]/a/p[2]')
        __p_3 = [ele.text for ele in _p_3]
        p_3 = ', '.join(__p_3)  # 분류를 하나의 문자열로 결합

        # 수집된 데이터를 리스트에 추가
        Com.append(p_2)
        Tit.append(p_1)
        Lin.append(p_0)
        Cat.append(p_3)
        
        elem_return_0('링크', Lin)
        elem_return_0('타이틀', Tit)
        elem_return_0('회사', Com)
        elem_return_0('역할', Cat)
        
    driver.quit()

    return Tit, Com, Lin, Cat



###########################################################################################



## 캐치 공고내용 수집
def collect_texts_from_iframes(Lin):
    # WebDriver 설정
    service = ChromeService(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service)
    
    Ctn = []

    for url in Lin:
        driver.get(url)
        
        # URL에서 숫자 추출
        match = re.search(r'RecruitInfoDetails/(\d+)', url)
        if match:
            number = match.group(1)
            
            # iframe이 로드될 때까지 기다림
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, f'//*[@id="view1_{number}"]/div/iframe'))
            )
            
            # iframe으로 컨텍스트 전환
            iframe = driver.find_element(By.XPATH, f'//*[@id="view1_{number}"]/div/iframe')
            driver.switch_to.frame(iframe)
            
            # iframe 내의 모든 텍스트 추출 및 리스트에 추가
            iframe_text = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            ).text
            Ctn.append(iframe_text)
            
            # 메인 컨텐츠로 컨텍스트 다시 전환
            driver.switch_to.default_content()
        else:
            Ctn.append("")
    
    # WebDriver 종료
    driver.quit()
    
    return Ctn



###########################################################################################



## 캐치 자격요건, 우대사항 추출
def extract_sections(text):
    # 문단이나 섹션의 시작을 나타내는 구분자 정의
    section_delimiter_pattern = r"(■|\●)"
    
    # 자격요건 관련 키워드 패턴
    requirements_pattern = r"(자격요건|지원자격|필수 경험과 역량|Required Skills|지원 자격)"
    # 우대사항 관련 키워드 패턴
    preferred_pattern = r"(우대사항|\[공통 우대 사항\]|채용하고 싶은 사람)"
    
    # 자격요건 및 우대사항 섹션 추출
    requirements_match = re.search(requirements_pattern, text, re.IGNORECASE)
    preferred_match = re.search(preferred_pattern, text, re.IGNORECASE)
    
    requirements_index = requirements_match.start() if requirements_match else len(text)
    preferred_index = preferred_match.start() if preferred_match else len(text)
    
    requirements_text = text[requirements_index:preferred_index].strip() if requirements_match else ""
    preferred_text = text[preferred_index:].strip() if preferred_match else ""
    
    # 우대사항 이후의 섹션 시작 구분자 찾기
    next_section_match = re.search(section_delimiter_pattern, text[preferred_index:], re.IGNORECASE)
    if next_section_match:
        preferred_text = text[preferred_index:preferred_index + next_section_match.start()].strip()
    
    return requirements_text, preferred_text



###########################################################################################



## 캐치 자격요건, 우대사항에서 기술스택툴 추출

def extract_tech_stacks_from_text(text, tech_stacks):
    found_stacks = []
    for stack in tech_stacks:
        if re.search(stack, text, re.IGNORECASE):
            found_stacks.append(stack)
    return ', '.join(found_stacks)



###########################################################################################



## 캐치 수집
def Catch():
    Tit, Com, Lin, Cat = collect_job_information()
    Ctn = collect_texts_from_iframes(Lin)
    
    # 데이터 프레임 생성
    df = pd.DataFrame({        
        'Title': Tit,
        'Company': Com,
        'Link': Lin,
        'label': Cat,
        'Content' : Ctn, 
        'platform': 'catch',
    })
    
    tech_stacks = [
    'Python', 'Java(?:Script)?', 'C\+\+', 'C#', 'PHP', 'Ruby', 'Swift', 'Kotlin', 'TypeScript', 'Scala', 'Go', 'Rust', 'Dart', 'R',
    'HTML', 'CSS', 'React(?:\.js)?', 'Angular(?:\.js)?', 'Vue(?:\.js)?', 'jQuery', 'Bootstrap', 'SASS', 'LESS', 'Node\.js', 'Express\.js',
    'Django', 'Flask', 'Spring Boot', 'Laravel', 'Ruby on Rails', 'ASP\.NET', 'Phoenix', 'TensorFlow', 'PyTorch', 'Keras', 'scikit-learn',
    'OpenCV', 'NLTK', 'Spacy', 'GPT(?:-\d+)?', 'BERT', 'FastAPI', 'GraphQL', 'RESTful API?', 'Apache Kafka', 'RabbitMQ', 'Docker', 'Kubernetes',
    'AWS', 'Google Cloud Platform', 'Azure', 'Firebase', 'MongoDB', 'PostgreSQL', 'MySQL', 'SQLite', 'Oracle', 'Microsoft SQL Server', 'Cassandra',
    'Redis', 'Elasticsearch', 'Git(?:Hub|Lab)?', 'Bitbucket', 'JIRA', 'Confluence', 'Slack', 'Microsoft Teams', 'Zoom', 'Jenkins', 'Travis CI',
    'CircleCI', 'Ansible', 'Terraform', 'Prometheus', 'Grafana'
    ]
    
    # 각 텍스트에 대해 "자격요건"과 "우대사항" 추출 및 새 컬럼에 저장
    df['Content_1'], df['Content_2'] = zip(*df['Content'].apply(extract_sections))

    # 각 텍스트에 대해 기술 스택 추출 및 새 컬럼에 저장
    df['Content_0'] = df['Content'].apply(lambda x: extract_tech_stacks_from_text(x, tech_stacks))

    keyword_csv_file = f'{path}/major.csv'
    df.to_csv(keyword_csv_file, index=False, encoding='utf-8-sig')
    
    return df



###########################################################################################



## st.button()    
# 빈 폴더 및 파일 생성
now = datetime.now()
now_name = now.strftime('%Y%m%d')
path = 'C:/Users/rlckd/Desktop/DS_Career/test-repo/src'
if not os.path.exists(path):
    os.makedirs(path)

# 버튼 클릭
KEYWORDS = ['데이터 분석가', '데이터 사이언티스트']
st.markdown('### 데이터 수집')
if st.button('크롤링 실행'):

    # 원티드 파일 제거
    if not os.path.exists(f'{path}/{now_name}_wanted.csv'):
        files_del = glob(f'{path}/*_wanted.csv')
        for file_del in files_del:
            os.remove(file_del)

        # 원티드 크롤링 시작 
        for KEYWORD in KEYWORDS:
            st_info = st.info(f'원티드 "{KEYWORD}" 크롤링 진행 중')
            Wanted(KEYWORD)

            time.sleep(1)
            st_info.empty()
            st_success = st.success(f'원티드 "{KEYWORD}" 크롤링 진행 완료')
            time.sleep(1)
            st_success.empty()

        # File Merge
        st_info = st.info('원티드 데이터 처리 중')
        DA = pd.read_csv(f'{path}/{KEYWORDS[0]}.csv')
        DS = pd.read_csv(f'{path}/{KEYWORDS[1]}.csv')
        df = pd.concat([DA, DS], axis=0)
        df.to_csv(f'{path}/{now_name}_wanted.csv', index=False, encoding='utf-8-sig')
        df = pd.read_csv(f'{path}/{now_name}_wanted.csv')

        # 원티드 Location 전처리 
        for i in range(0, len(df)):
            s_0 = df.Location[i].split(' · ')
            df.Location[i] = s_0[0]
        df.to_csv(f'{path}/{now_name}_wanted.csv', index=False, encoding='utf-8-sig')

        time.sleep(1)
        st_info.empty()
        st_success = st.success('원티드 데이터 처리 완료')
        time.sleep(1) 
        st_success.empty()

        # File Remove
        os.remove(f'{path}/{KEYWORDS[0]}.csv')
        os.remove(f'{path}/{KEYWORDS[1]}.csv')
    else:
        st_info = st.info('생성된 파일이 있습니다: 원티드')
        time.sleep(1)
        st_info.empty()

    # 캐치 파일 제거
    if not os.path.exists(f'{path}/{now_name}_major.csv'):
        files_del = glob(f'{path}/*_major.csv')
        for file_del in files_del:
            os.remove(file_del)
        
        # 캐치 크롤링 시작
        st_info = st.info('네카라쿠배당토 크롤링 진행 중')
        Catch()

        time.sleep(1)
        st_info.empty()

        # File Merge
        st_info = st.info('네카라쿠배당토 데이터 처리 중')
        MAJOR = pd.read_csv(f'{path}/major.csv')
        MAJOR.to_csv(f'{path}/{now_name}_major.csv', index=False, encoding='utf-8-sig')
        df = pd.read_csv(f'{path}/{now_name}_major.csv')

        time.sleep(1)
        st_info.empty()
        st_success = st.success('네카라쿠배당토 데이터 처리 완료')
        time.sleep(1) 
        st_success.empty()

        # File Remove
        os.remove(f'{path}/major.csv')
    else:
        st_info = st.info('생성된 파일이 있습니다: 네카라쿠배당토')
        time.sleep(1)
        st_info.empty()
