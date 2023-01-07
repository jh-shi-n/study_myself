# Intro

- 파이썬에서 반복적인 작업을 자동화하기 위해 조직적으로 관리하는 가장 일반적인 방법은 `List`
- 이에 보완할 수 있는 데이터 타입 : `dict`
    - 검색에 사용할 키와 키에 연관된 값을 저장한다.
        - 해시 테이블 (Hash Table)
        - 연관 배열 (Associative Array)
    - 상수 시간에 원소를 삽입하고 찾을 수 있다.
    - 동적인 정보를 관리하는 데는 `dict` 가 일반적
- 파이썬은 리스트와 딕셔너리를 다룰 때 가독성을 좋게 하고, 기능을 확장해주는 특별한 구문과 내장 모듈을 제공


---
### Better Way 11. 시퀀스를 슬라이싱하는 방법을 익혀라

- 슬라이싱의 기본 형태 **리스트[시작:끝]**
- 시작 인덱스에 있는 원소는 슬라이스에 포함, 끝 인덱스 원소는 포함되지 않음

```python
a = ['a' , 'b',  'c',  'd', 'e', 'f', 'g', 'h']

print('가운데 2개',  a[3:5])
print('마지막 제외한 나머지' , a[1:7])
```

- 리스트의 맨 앞부터 슬라이싱할때는 시각적 잡음을 없애기 위해 0 생략
- 리스트의 끝까지 슬라이싱할때는 쓸데없이 끝 인덱스를 적지 않음

```python
# Good Case
a[:5]
a[5:]
# Bad Case
a[0:5]
a[5:len(a)]
```

- 리스트의 끝에서부터 원소를 찾고싶을때는 음수 인덱스 사용
    
    **a = ['a' , 'b',  'c',  'd', 'e', 'f', 'g', 'h']**
    
    a[:-1] = ['a' , 'b',  'c',  'd', 'e', 'f', 'g']
    
    a[-3:] = ['f', 'g', 'h']
    
    a[2:-1] = [ 'c',  'd', 'e', 'f', 'g' ]
    
- 리스트를 슬라이싱한 결과는 완전히 새로운 리스트

b = a[2:3]

- *리스트에 지정한 슬라이스 길이보다 대입되는 배열의 길이가 짧을 경우, 리스트 감소*
- *리스트에 지정한 슬라이스 길이보다 대입되는 배열의 길이가 길 경우, 리스트 증가*

*a[2:7] = [1,2,3]*

*a = [’a’,’b’,1,2,3’,’h’]*

*a[2:3] = [47,11]*

*a = [’a’,’b’,47,11,2,3,’h’]*

- 리스트 내 2칸 씩 움직이게 하기 위해서는 :: 사용

a[::2]

### Better way 12) 스트라이드와 슬라이스를 한 식에 함께 사용하지 말라

- 파이썬은 일정한 간격을 두고 슬라이싱 할 수 있는 특별한 구문(스트라이드)을 제공

```python
x = [1, 2, 3, 4, 5, 6]
odds = x[::2]
evens = x[1::2]
print(odds)
print(evens)
"""
[1, 3, 5]
[2, 4, 6]
"""
```

- 일반적으로 문자열을 스트라이드를 활용해서 역으로 뒤집을 수 있음
- 유니코드(2바이트) 데이터를 UTF-8로 인코딩한 문자열에서는 동작하지 않음

```python
x = "abcdef"
print(x[::-1])
"""
fedcba
"""

w = "나는 행복합니다" # 영어(1바이트)는 정상동작함
x = w.encode("utf-8")
y = x[::-1]
z = y.decode("utf-8")
print(z)
"""
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa4 in position 0: invalid start byte
"""
```

- 되도록 시작값이나 끝값을 증가값과 함께 사용하지 않거나, 사용할 경우 슬라이싱과 스트라이딩을 분리시킬 것
- 프로그램이 두 단계 연산에 필요한 시간과 메모리를 감당할 수 없다면 `itertools`의 `islice`함수를 고려(Better way 36 참고)

```python
# bad case
y = x[1:-1:2]

# better case
y = x[::2]
z = y[1:-1]
```

### Better way 13 슬라이싱 보다는 나머지를 잡아내는 언패킹을 사용해라

- 앞서 언급 되었던 `언패킹` 을 활용하는 것의 한계점은 언패킹할 시퀀스의 길이를 미리 알고있어야 한다.
- 하단의 코드처럼 작성하는 경우, 시각적인 잡음도 많고, 인덱스로 인한 오류를 발견하기 쉽다.

```python
car = [2,1,3,5,4,6]
car_ages_descending = sorted(car, reverse=True)

oldest = car_ages_descending[0]
second_oldest = car_ages_descending[1]
others = car_ages_descending[2:]

print(oldest, second_oldest, others)
```

- 이를 보완하여, 별표 식을 이용한다면 위와 동일하게 작동하지만, 오류 발생의 가능성을 없앨 수 있음

```python
oldest, second_oldest, *others = car_age_descending
```

- 하지만, 해당방식으로 작동시키며 이터레이터를 활용하는 경우, 컴퓨터 메모리를 모두 활용하여, 프로그램이 멈출수도 있음
- 결과 데이터가 모두 메모리에 들어갈 수 있다고 생각하는 경우에만 활용할 것

### Better way 14 복잡한 기준을 사용해 정렬할 때는 key 파라미터를 사용해라

- list 에서는 sort 사용가능

```python
numbers = [11,93,61, 68, 70]
numbers.sort()
```

- 복잡한 리스트에 대해서는 사용불가

### Better way 15 딕셔너리 삽입 순서에 의존할때는 조심하라

- 파이썬 3.5버전 이전에서는 딕셔너리에 대해 이터레이션 수행 시, 키를 임의의 순서로 돌려줌
- 파이썬 3.6버전 이후부터는 딕셔너리가 삽입 순서를 보존하도록 동작이 개선

```python
baby_names = {
			'cat' : 'kitten',
			'dog' : 'puppy'
							}

# python 3.5
print(list(baby_names.keys()) -> ['dog', 'cat']
print(list(baby_names.values())  -> ['puppy', 'kitten']
print(list(baby_names.items())
print(list(baby_names.popitem())

# after
print(list(baby_names.keys()) -> ['cat', 'dog']
print(list(baby_names.values()) -> ['kitten', 'puppy']
print(list(baby_names.items())
print(list(baby_names.popitem())
```

### Better way 15) 딕셔너리 삽입 순서에 의존할 때는 조심하라

- python 3.5 이전에는 딕셔너리에 대해 이터레이션을 수행하면 키를 임의로 sort해서 돌려줬음 → 원소가 삽입된 순서랑 다름
- 이유는 딕셔너리 구현이 내장 hash함수와 파이썬 인터프리터가 시작할 때 초기화되는 난수 seed값을 사용하는 해시테이블 알고리즘으로 만들어졌기 때문 → 호출할 때 마다 key의 순서가 바뀌기 때문에 sort가 필수적
- 단순히 python 3.6 이후에는 dict가 삽입 순서에 의존하기 때문에 그것을 기준으로 함수를 짜지 말고, sorted dict와 같이 dict를 커스텀해서 사용하는 경우도 고려하여서 조심해서 함수를 짤 것

### Better way 16) in을 사용하고 딕셔너리 키가 없을 때 Keyerror를 처리하기보다는 get을 활용하라

- 딕셔너리와 상호작용하는 연산
    - 키 & 키 연관 값에 접근
    - 대입
    - 삭제
- 간단한 딕셔너리의 경우, get 메서드를 사용하면 매우 간단해진다
- 복잡한 딕셔너리의 경우, get 메서드를 사용해도되지만,  defaultdict를 사용해도 좋다
    - get과 동일한 기능인 setdefault가 있긴하지만, 에러 발생이 잦음
    - 딕셔너리에 키가 없는 경우, setdefault에 전달된 디폴트값이 별도로 복사되지않고, 딕셔너리에 대입됨.

```python
## 간단한
key = '밀'
if key in counters:
	count = counters[key]
else:
	count = 0

counters[key] = count + 1

#get
count = counters.get(key, 0)
counters[key] = count + 1

## 복잡한
votes = {
'A' : {'a','b'}
'B' : {'c','d'}
}

key = 'C'
who = 'e'

if key in votes:
	names = votes[key]
else:
	votes[key] = names = []

```

### Better way 17 내부상태에서 원소가 없는 경우를 처리할때는 setdefault보다 defaultdict를 사용하라

- 사용 용도에 따라 get, setdefault, defaultdict를 유동적으로 활용하자

```python
# 방문했던 지역의 이름을 저장하고 싶다.
visit = {
"미국" : {'뉴욕','로스앤젤레스'}
"일본" : {'하코네'}
}

# 딕셔너리 안에 나라 이름 존재 여부와 관계없이 각 집합에 새 도시를 추가할 경우, 다음과 같이.
# 서로 동일하게 작동함
visits.setdefault('프랑스',set()).add('칸')

*if (japan := visit.get('일본')) is None:*
	visits('일본') = japan = set()
japan.add('교토')

# 딕셔너리 생성을 제어할 수 있다면?
Class visits:
	def __init__(self):
		self.data = {}

	def add(self, country, city):
		city_set = self.data.setdefault(country, set()) # 함수 이름이 헷갈림
		city_set.add(city)

	#개선하면?
Class visits:
	def __init__(self):
		self.data = defaultdict(set)

	def add(self, country, city):
		self.data[country].add(city)
```

### Better way 18  _ _missing_ _ 을 사용해 키에 따라 다른 디폴트 값을 생성하는 방법을 알아두라
- `open`을 통해 파일을 호출하는 경우 key값으로 호출하는 방법이 있음
- 코드를 간결하게 하기 위해 `open`을 하고 `OSError`을 예외처리하는 방법도 있지만, 매번 `open` 메서드가 호출되고 기존에 열린 파일핸들과 혼동될 수 있음
- 또한, `OSError`를 통해 예외처리를 하면 같은 줄의 `setdefault`가 던지는 예외와 구분하지 못하는 경우도 발생
- defaultdict를 사용할 수도 있지만 defaultdict의 생성자에 전달한 함수는 인자를 받을 수 없음
- 이 경우 dict 타입의 하위클래스를 만들고 __missing__메서드를 구현해서 커스텀화 할 수 있음

```python
# bad case
pictures = {}
path = "profile_1234.png"

if (handle := pictures.get(path)) is None:
	try:
		handle = open(path, "a+b")
	except OSError:
		print(f"경로를 열 수 없습니다: {path}")
		raise
	else:
		picture[path] = handle

handle.seek(0)
image_data = handle.read()

# worse case
try:
	handle = pictures.setdefault(path, open(path, "a+b"))
except OSError:
	print(f"경로를 열 수 없습니다: {path}")
		raise
	else:
		handle.seek(0)
		image_data = handle.read()

# error case
from collections import defaultdict

def open_picture(profile_path):
	try:
		return open(profile_path, "a+b")
	except OSError:
		print(f"경로를 열 수 없습니다: {profile_path}")
		raise

pictures = defaultdict(open_picture)
handle = pictures[path]
handle.seek(0)
image_data = handle.read()
"""
TypeError: open_picture() missing 1 required positional argument: 'profile_path'
"""

# better case
class Pictures(dict):
	def __missing__(self, key):
		value = open_picture(key)
		self[key] = value
		return value

pictures = Pictures()
handle = pictures[path]
handle.seek(0)
image_data = handle.read()
		
```
