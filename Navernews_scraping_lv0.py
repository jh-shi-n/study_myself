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
   
#네이버 검색창에 Search_value 입력값을 검색 후, 결과로 나오는 뉴스 결과 1페이지 정보를 얻어옴
