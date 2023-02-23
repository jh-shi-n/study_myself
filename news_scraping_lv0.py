import requests
from bs4 import BeautifulSoup


##가장 기본 형태
#검색어 입력 (string 입력)
search_value = input(str())

#f-string 형식으로 진행
raw = requests.get(f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={search_value}")
html = BeautifulSoup(raw.text, "html.parser")
article = html.select('ul.list_news > li')

#개발자 도구로 확인시, news_tit에 기사 제목이 들어있었음
for k in article:
    title = k.select_one('a.news_tit').text
    print(title)
   
#네이버 검색창에 Search_value 입력값을 검색 후, 결과로 나오는 뉴스 결과 1페이지 기사 Title을 얻어옴


#-----------------------------------------------------------------------------------------------------------

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

#검색어 입력 (string 입력)
search_value = input(str())
sort_value = input()
# 0 = 관련성, 1= 최신, 2=오래된순

# 기본설정
result = pd.DataFrame()
page_number = 1

#range 값 1 증가마다 10개씩 더 긁어옴
for time in range(0,4):
    
    url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search_value}&sort={sort_value}&photo=0&field=0&pd=0&ds=&de=&mynews=1&office_type=1&office_section_code=2&news_office_checked=1001&nso=so:dd,p:all,a:all&start={page_number}"
    headers = {'user-agent':'Mozilla/5.0'}
    raw = requests.get(url, headers=headers)    
    html = BeautifulSoup(raw.text, 'lxml')
    
    #기사 정보
    news_title = [title['title'] for title in html.find_all('a', attrs={'class':'news_tit'})]
    news_url = [url['href'] for url in html.find_all('a', attrs={'class':'news_tit'})]
    news_date = [date.get_text() for date in html.find_all('span', attrs={'class':'info'})]
    
    df = pd.DataFrame({'기사제목' : news_title, '기사작성' : news_date, "기사주소" : news_url})
    result = pd.concat([result, df])
    
    page_number += 10

result
