#Titanic: Machine Learning from Disaster)

predict survival on the Titanic using Excel, Python, R & Random Forests.

문제링크 : [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)


 - 사용 언어 : python
 - 사용 라이브러리 : pandas, scikit-learn
 
---

**RMS 타이타닉**은 1912년 4월 10일 영국의 사우스햄튼에서 출항하여 미국의 뉴욕으로 항해하던 첫 항해 중에 4월 15일 빙산과 충돌하여 침몰하였다. 타이타닉의 침몰로 1,541명이 사망하였으며, 이는 평화시 해난 사고 가운데 가장큰 인명피해가운데 하나이다.
[https://ko.wikipedia.org/wiki/RMS_타이타닉](https://ko.wikipedia.org/wiki/RMS_타이타닉)

kaggle의 getting started competition중 하나인 타이타닉 문제이다. 데이터는 [링크](https://www.kaggle.com/c/titanic/data)에서 다운받을 수 있으며 데이터에는 승객과 관련하여 다음과 같은 항목이 기록되어있다.

  - ```PassengerId``` 각 승객의 고유 번호
  - ```Survived``` 승객의 생존 여부 생존(1), 사망(0)
  - ```Pclass``` 승객이 속한 등급 (1)등급 (2)등급 혹은 (3)등급
  - ```Name``` 승객의 이름
  - ```Sex``` 승객의 성별
  - ```Age``` 승객의 나이
  - ```SibSp``` 승객과 함께 탑승한 형제자매와 배우자의 수
  - ```Parch``` 승객과 함께 탑승한 자녀의 수
  - ```Ticket``` 승객의 티켓넘버
  - ```Fare``` 운임
  - ```Cabin```짐의 위치
  - ```Embarcked``` 탑승한 입구
  
먼저 승객의 어떤 그룹이 더 많이 생존했는가 생각을 해보아야한다. 문제의 [description](https://www.kaggle.com/c/titanic)을 보면 
  ```some groups of people were more likely to survive than others, such as women, children, and the upper-class.```
  
여성, 어린이, 그리고 상위클래스가 많이 생존했다는 사실을 확인할 수 있다. 이 점에서 시작하여 문제를 풀어본다.

---

각 column에 대한 특징을 살펴보기 위해 pandas 라이브러리의 ```.describe()```함수를 실행해 본다. 

수행하고 나면 각 column의 count보다 작은 column이 있는데 이는 해당 column의 어떤 데이터가 빠진것이다. 이 경우 직접 데이터를 채워넣어야 한다. 여러 방법이 있겠지만 중간값을 넣는것으로 데이터를 채워넣는다.

    .fillna(COLUMN.median())

```.describe()```함수를 수행했을 때 보이지 않았던 column들이 있었다. 숫자가 아닌 non-numeric column들을 숫자로 바꿔준다.


이제 test set에 대한 linear regression 알고리즘을 사용 할 수 있다.

overfitting을 피하고, 예측값을 만들기 위해서 알고리즘을 다른 데이터셋에 적용하는것이 좋다. 혹은 ```cross validation```을 사용하여 ovefitting을 피하는 법이있다. 

전처리가 끝났다면 이제 ```skit-learn``` 라이브러리를 사용하여 예측값을 만든다. 첫 번째로 만든 예측값은 정확도 78%로 좋지만 훌륭한 수준은 아니다. 따라서 개선을 해야한다. 방법이 여러가지가 있는데
 
 - 더 좋은 머신러닝 알고리즘을 사용한다.
 - 더 나은 특징들을 만들어낸다.
 - 여러 머신러닝 알고리즘을 조합한다.
 
위 3가지 방법으로 결과를 개선한다.

 
 
--- 
##알고리즘 개선

  - 더 좋은 알고리즘 : random forest
  
첫번째 방법은 더 좋은 머신러닝 알고리즘 적용으로 [random forest](https://ko.wikipedia.org/wiki/%EB%9E%9C%EB%8D%A4_%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8)알고리즘을 사용한다.

 
	from sklearn.ensemble import RandomForestClassifier

Random forest 알고리즘의 정확성을 향상하기 위해 사용하는 tree의 갯수를 증가시킨다. tree를 학습시키는데 더 오랜시간이 걸리지만 정확성을 향상시킬 수 있다. 
Random forest 알고리즘을 수행하면서 overfitting이 발생할 수 있으므로 이를 방지하기위해 ```min_samples_split```과 ```min_samples_leaf```를 수정한다. 

 - 특징 더 만들기

지금까지는 나이와 성별로만 데이터를 만들었지만 다른 특징을 만들기 위해서 특징들을 조합해본다.

 - ```이름의 길이``` 이름이 길수록 지위나 재산이 많음을 유추할수 있다. 
 - ```함께 승선한 가족의 수``` (```SibSp``` + ```Parch```)
 - ``title`` Mr, Mrs, Miss등

특징을 만드는 가장 쉬운방법은 ```.apply``` pandas dataframe에 method를 사용하는 것이다. 이 method는 dataframe과 series에 속한 각각의 element들은 전달해준다. 우리는 ```lambda```함수를 이용한다.

	# Generating a familysize column
	titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

	# The .apply method generates a new series
	titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
	


```title```을 이용해 보자.

먼저 ```title```을 이용하기 위해서는 이름에서 타이틀을 추출해 내야햔다. 보통 ```title```은 ```Mr.```, ```Mrs.```등이 널리 쓰이지만 종종 다른 경우가 있는것을 데이터를 보면 확인 할 수 있다. 이런 경우는 1~2가지에 불과하므로 무시한다. 먼저 ```title```은 ```non-numberic```이므로 numberic으로 바꿔주는 작업을 수행한다.


그 다음 ```가족 단위```을 활용 할 수 있다.

이름의 성을 살펴보면 이 사람이 어떤 가족에 속해있는지 알 수 있다. 이를 위해 ```Faimily```size와 이름의 성을 가지고 처리한다.


가장 좋은 특징을 찾아내자.

```Feature engineering ```은 머신러닝에서 가장 중요한 부분이다. 각각의 데이터에는 계산할 수 있는 많은 양의 데이터 특징이 있다. 각 특징을 찾아내는 것도 중요하지만 가장 좋은 특징을 찾아내는것 또한 중요하다.
가장 좋은 특징을 찾는 방법중 하나는``` univariate feature selection```을 이용하는 것이다. 이는 각각의 column을 살펴본 후 어떤 column이 예측하고자 하는 (이 문제에서는 ```survived```) prediction과 가장 연관이 있는지 찾을수 있다.
```skit-learn``` 라이브러리의 ```SelectKBest```를 사용하면 쉽게 구현 할 수 있다.

descision tree를 구현하는 다른 방법중 하나는 ```gradient boosting tree```를 이용하는 것이다. 


---

## 정확성을 향상시키는 법(진행중)

 - feature engineering
	 - cabin과 관련된 특성을 이용한다.
	 - 가족구성원의 크기 : 가족의 여성의 숫자가 가족 전체를 살릴 수 있을까
	 - 승객의 이름으로 출신을 알 수 있을것이다. 이게 생존과 연관이 있을까? 
	 
  - 알고리즘
    - random forest classfier
    - support vector machine
    - 다른 base classifier
    
  - ensembling methods
    - majority voting
    




---
##용어정리

 - overfitting
 
## reference

 - [regular expression](https://en.wikipedia.org/wiki/Regular_expression)
 - [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)



