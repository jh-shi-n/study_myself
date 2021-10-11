#기본 틀

nicknames = "~닉네임~"
api_key = "~API~"

#API는 header로 입력
headers = {"Authorization" : api_key}

api_url = f'https://api.nexon.co.kr/kart/v1.0/users/nickname/{nicknames}'
status_url = f'https://api.nexon.co.kr/kart/v1.0/users/{user_id}'
match_url = f'https://api.nexon.co.kr/kart/v1.0/users/{user_id}/matches'

#닉네임 입력 받고 유저정보 검색
res = requests.get(api_url, headers = headers)
user_id = res.json()['accessId']

res3 = requests.get(status_url, headers = headers)
stat = res3.json()



if ('name' in stat) :
    user_frame = pd.DataFrame([stat['accessId'], stat['name'], stat['level']], index = ['고유코드','닉네임', '레벨'])
    print(user_frame)
else:
    print('오류')
