# Day3

[1. 자료형 이해](#-스택과-큐)

[2. What is Pythoic?](#-Pythonic-Code)

[3. generator and advanced function](##generator)

# | 개인 학습

:sparkles: 정규식에 대해서 추가로 공부했습니다. 얕게 공부한 느낌이라 더 깊이 공부해야 할 것 같습니다.

https://blog.naver.com/sjy263942


# | 회고
오늘 강의는 파이썬 코드에 대해서 깊이 생각해볼 수 있도록 도와준 강의였습니다. 피어 세션에서도 과제 때 제출한 코드를 리뷰하면서, 제가 고쳐야 할 버릇은 무엇인지 비효율적인 부분은 어떻게 개선해야 하는지 많이 배웠습니다. 파이썬이 많이 익숙하더라도 좋은 코드가 무엇인가에 대해서는 끊임없이 고민해야 할 것 같습니다.

# 


# 스택과 큐

* 스택	

1.  LIFO 구조
2. 데이터 Push, Pop

* 큐

1. FIFO - 먼저 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조

# 튜플과 집합

* 튜플

0. 쓰는 이유 - 프로그램 작동 중 변경되지 않은 데이터의 저장. 함수 반환값 등 사용자의 실수에 의한 에러 사전에 방지
1. 값의 **변경이 불가능**한 리스트 (불변)
2. 선언시 [] 가 아닌 () 사용
3. 리스트의 연산, 인덱싱, 슬라이싱 동일하게 사용
4. 값이 하나인 튜플은 반드시 (1,) 붙여야함

* 집합

1. 값을 순서없이 저장, 중복 불허하는 자료형
2. set 객체 선언 이용해 객체를 생성
3. s1.intersection(s2) : s1과 s2의 교집합
    s1.difference(s2) : s1과 s2의 차집합

# 사전

* 데이터 저장할 때 구분 지을 수 있는 값 함께 저장. value를 고유한 값으로 정해줄 수 있도록 key가 있다.

# Collection 모듈

* 리스트, 튜플, 딕트에 대한 파이썬 빅트인 확장 자료구조 모듈
* 편의성, 효율(메모리 사용량 줄이고, 시간 빠르게) 사용자에게 제공

* 주요

1. deque : 스택과 큐를 지원하는 모듈. 리스트에 비해 빠른 자료 저장 방식을 지원
    * rotate, reverse 등 `Linked List`의 특성을 지원
    * 기존 리스트 형태의 함수를 모두 지원함
    - [x] Linked List
    값들이 순차적으로 메모리에 연결되는 것이 아니라, 다음 데이터가 저장된 주소를 계속 가리키면서 유연하게 시퀀스 데이터를 표현하는. 

2. OrderedDict 
    * Dict와 달리 데이터를 입력한 순서대로 dict를 반환함
    * 그러나 python 3.6부터는 입력한 순서 보장하여 출력해서 의미는 없다.

3. defaultdict
    * Dict type의 값에 기본 값을 지정, 신규값 생성시 사용 - keyError 방지
    * 모든 키에 대해서 0이 값으로 저장됨 - (jy)이거 코테때 조심해야

4. Counter
    * 시퀀스 타입의 데이터 원소 갯수를 dict형태로 반환
    * Set 연산들을 지원함

    ```python
    from collections import Counter
    
    c=Counter(a=4,b=2,c=0,d=-2)    
    d=Counter(a=1,b=2,c=3,d=4)
    
    print(c+d)
    print(c&d) # 교집합
    print(c|d) # 합집합
    ```

5. namedtuple

    * 튜플 형태로 데이터 **구조체를 저장**하는 방법
    * 저장되는 데이터의 variable을 사전에 지정해서 저장

    ```python
    from collections import namedtuple
    
    Point=namedtuple('Point',['x','y']) # 이 튜플의 이름은 point
    p=Point(x=11,y=22) # Point 구조체에서 x와 y를 지정
    x,y=p
    
    # same
    print(x,y)
    print(p.x,p.y)
    print(Point(x=11,y=22))
    ```

    

# Pythonic Code

0. 왜 쓰는지? - **남의 코드에 대한 이해도를 높이기** 위해서. 많은 개발자들이 파이썬 스타일로 코딩한다. for루프보다 리스트가 조금 더 빠르다 이런 것 이해해야. 익숙해지면 코드도 짧아진다. 마지막으로 멋있다.
1. 파이썬 스타일의 코딩 기법
2. 파이썬 특유의 문법을 활용해 효율적 코드를 표현
3. 그러나 더 이상 파이썬 특유의 스타일은 아님. 많은 언어들이 서로의 장점을 채용
4. 고급 코드를 작성할 수록 더 많이 필요해짐


## split & join

## list comprehension

* 기존 리스트를 사용해 다른 리스트를 만드는 기법
* 포괄적인 리스트, 포함되는 리스트라는 의미로 사용됨
* 파이썬에서 가장 많이 사용되는 기법
* 일반적으로 for+append보다 속도가 빠름.
`result=[i+j for i in word1 for j in word2 if not (i==j)]`
`result=[i+j if not (i==j) else i for i in word1 for j in word2]`
* two dimensional   
`result=[[i+j for i in case1] for j in case2]`
**위 코드는 아래와 같음**
```python
for j in case2:
line=[]
for i in case1:
    line.append(i+j)
```

## enumerate & zip
* enumerate : 리스트의 element를 추출할 때 번호를 붙여서 추출. idx, value

* zip : 두개의 리스트 값을 병렬적으로 추출함. 각 튜플을 같은 인덱스끼리 묶는 등
```python
a=[sum(x) for x in zip((1,2,3),(10,20,30),(100,200,300))]
# a==[111,222,333]
```
```python
alist=['a1','a2','a3']
blist=['b1','b2','b3']

print(enumerate(zip(alist,blist)))
# [(0,('a1','b1')),(1,('a2','b2')),(2,('a3','b3'))]
```

## lambda & map & reduce
* lambda : 함수 이름 없이 함수처럼 쓸 수 있는 익명 함수
    * 수학의 람다 대수에서 유래함
    * PEP8에서는 권장하지 않음 - 테스트의 어려움, docstring 지원 미비, 코드 해석 어려움... 그래도 많이 쓴다.

* map : 시퀀스 데이터에 어떤 함수를 mapping해줌
    * 실행 시점의 값을 생성하므로 메모리 효율적
    * 그러나 최근에는 권장되지 않음. 해석하기 어렵다. list comprehension으로 표현하는게 훨씬 보기 좋다. .. 그래도 많이 쓴다.
    ```python
    def f(x):
        return 

    ex=[1,2,3,4,5]
    print(list(map(f,ex))) # generator 반환하므로 리스트로 감싸줘야
    ```
* reduce : map 함수와 달리 list에 똑같은 함수를 적용해서 통합
    ```python
    from functools import reduce
    print(reduce(lambda x,y:x+y, [1,2,3,4,5]))
    # 3,6,10,15 => 15 최종출력
    ```

## generator
* iterable object의 특수한 형태로 사용하는 함수
* iterable object : 시퀀스 자료형에서 데이터를 순서대로 추출하는 object
    * 내부적으로 __inter__와 __next__가 사용된다. 따라서 iter(), next() 함수로 iterable한 객체롤 iterator object로 사용한다.
    ```python
    cities=["a",'b','c']
    memory_address=iter(cities)
    print(memory_address)

    iter_obj=iter(cities)
    print(next(iter_obj)) # 다음 위치만 알고 있다.
    print(next(iter_obj))
    print(next(iter_obj))
    ```
* element가 사용되는 시점에 메모리 값을 반환. 그 전에는 주소값만 가지고 있음. "여기 저장되어 잇대!"
* yield를 사용해 한번에 하나의 element만 반환한다.
```python
import sys

def general_list(value):
    res=[]
    for i in range(value):
        res.append(i)
    return res

def generator_list(value):
    res=[]
    for i in range(value): # 값을 메모리에 안올리고 주소값만 가지고 있다가 값달라고 하면 yield가 하나씩 던져줌 (list로 감싸면 다 메모리에 올려줌)
        yield i

print(sys.getsizeof(general_list(50))) #264 바이트
print(sys.getsizeof(generator_list(50))) # 64 바이트 - 메모리 절약
```
* generator comprehension
   1. list 컴프리헨션과 유사한 형태로 generator형태의 list생성
   2. generator expression이라는 이름으로도 부름
   3. [] 대신 ()를 사용해 표현
   `gen_Ex=(n*n for n in range(50))`
* 메모리 아껴야 하는 큰 데이터/파일 데이터 처리할 때 제너레이터 써라!


## function passing arguments
* 함수에 입력되는 arguments의 다양한 형태
   1. keyword arguments
        `f(name={name},child={child})`
   2. default arguments
        `f(name={name},child="child1")`
   3. variable-length asterisk
        * 개수가 정해지지 않은 변수를 함수의 parameter로 사용하는 법
        * keyword arguments와 함께 argument추가가 가능
        * asterisk(*) 기호를 사용해 함수의 parameter를 표시함 
        * 입력된 값은 tuple type으로 사용할 수 있음
        * 가변 인자는 오직 한개만 맨 마지막 parameter위치에 사용 가능
        ```python
        def asterisk_test(a,b,*args):
            return a+b+sum(args) # unpack,pack 다 가능
        print(asterisk_test(1,2,3,3,4,5)) # 15
        ```
    4. Keyword variable length
    * 파라메터 이름을 지정하지 않고 입력하는 방법
    * ** 로 asterisk 두개 사용해 함수의 파라미터 표시
    * 입력된 값은 dict type으로 사용할 수 있다
    * 가변 인자는 오직 한개만 기존 가변인자 다음에 사용
    ```python
    def kwargs_test_1(**kwargs):
        print(kwargs)

    kwargs_test_1(first=3,second=4,third=5)     
    ```

    ```python
    def kwargs_test_2(one,two,*args,**kwargs):
        print(one)
        print(two)
        print(args) # tuple
        print(kwargs) # dict

    kwargs_test_2(3,4,5,6,7,8,9,first=3,second=5)
    ```

## asterisk
* 단순 곱셈, 제곱 연산, 가변 인자 활용 등 다양하게 사용됨
* unpacking 기능 많이 씀
    - [x] unpacking a container
    - tuple,dict 등 자료형에 들어가 있는 값을 unpacking
    - 함수의 입력값, zip등에 유용하게 사용
    ```python
    # args앞에 붙은 *는 pack, *()으로 붙은 *는 unpack
    def asterisk_test(a,*args):
        print(a,args)

    asterisk_test(1,(2,3,4,5,6)) # 1 ((2, 3, 4, 5, 6),)
    asterisk_test(1,*(2,3,4,5,6)) # 1 (2, 3, 4, 5, 6)

    def asterisk_test2(a,b,c):
        print(a,b,c)
    data={'b':1,'c':1}
    asterisk_test2(10,**data) # **는 dict unpack
    ```
    ```python
    print(*[1,2,3,4]) # 1 2 3 4
    ```
* with zip
```python
ex=([1,2],[3,4])
for value in zip(*ex):
    print(value)
```