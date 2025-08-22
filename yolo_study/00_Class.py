# ------------------------------------------------------
# 클래스 기초
# ------------------------------------------------------
import json
import csv


class BankAccount:  # 클래스의 인스턴스 생성이 목표
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):  # 인스턴스 메소드(함수)
        self.balance += amount
        return self.balance

    def withdraw(self, amount):  # 인스턴스 메소드
        if amount > self.balance:
            return "잔액 부족"
        self.balance -= amount
        return self.balance


# 인스턴스 생성 및 사용
account = BankAccount("홍길동", 1000)
# 1. 인스턴스 생성, 2. 인스턴스 초기화, 3. 참조 리턴

print(account.deposit(500))
print(account.withdraw(200))

account = BankAccount("Scott")
print(account.deposit(1000))
print(account.withdraw(500))


# ------------------------------------------------------
# 상속시에 자식 클래스에서 부모 클래스의 속성을 초기화하는 예
# ------------------------------------------------------
class Parent:
    def __init__(self, name):
        self.name = name
        print(f"Parent 생성자 호출: name = {self.name}")


class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # 부모 생성자 호출
        # super()는 self를 랩핑하여 기능확장하여 부모 클래스의 메소드 호출 가능하도록 한 Proxy 오브젝트
        self.age = age
        print(f"Child 생성자 호출: age = {self.age}")


# 테스트
child = Child("홍길동", 20)


# ------------------------------------------------------
# CSV, JSON 파일 실전
# Comma Separated Value, Javascript Standard Object Notation
# ------------------------------------------------------

# CSV 저장
data = [["이름", "나이"], ["홍길동", 30], ["김영희", 25]]

file_path = "/content/drive/MyDrive/Python_AI/YOLO/Codes/Python Advanced/"

with open(file_path+"people.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# CSV 읽기
with open(file_path+"people.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        name, age = row                   # unpack
        print(f"{name}\t{age}")           # f-string
        # print("{}\t{}".format(name,age)) # format()
        # print("%s\t%s"%(name,age))       # %s

# JSON 저장
person = {"name": "홍길동", "age": 30, "email": "hong@example.com"}
with open("person.json", "w", encoding="utf-8") as f:
    json.dump(person, f, ensure_ascii=False, indent=2)

# JSON 읽기
with open("person.json", "r", encoding="utf-8") as f:
    person_data = json.load(f)
    print(person_data)


'''

클래스, 파일 스트림, 컨테이너 종합실습
기본적인 CRUD(Create, Read, Update, Delete, 검색)
키보드 입력
사원정보 관리 시스템
추가(a), 목록(s), 수정(u), 삭제(d), 검색(f), 종료(x) :
a : 이용자로부터 사번, 이름, 부서번호, 전화번호 입력 및 csv 파일에 추가
추가할 때는 파일 모드를 "a" 로 지정, "w":덮어쓰기, "r":읽기, "a":append
파일명 : employee.csv
s : 사원 목록을 화면에 표시
u : 수정할 사번과 새 전화번호를 입력 받아서 기존 데이터 변경
기존 데이터를 모두 로드하여 리스트에 저장하고 수정대상 정보를 찾아 변경
메모리에서 변경된 데이터를 다시 employee.csv 파일에 덮어쓰기
d : 삭제대상 사번을 입력 받아서 해당 사원 정보 삭제
기존 데이터를 모두 로드하여 리스트에 저장하고 수정대상 정보를 찾아 삭제
메모리에서 삭제된 데이터를 다시 employee.csv 파일에 덮어쓰기
f : 검색하려는 사원 번호를 입력하여 검색된 사원 정보를 화며에 표시
x : 프로그램 메인 루프 종료
'''

# ------------------------------------------------------


class Employee:
    def __init__(self, info):
        eno, ename, dno, phone = info
        self.eno = eno
        self.ename = ename
        self.dno = dno
        self.phone = phone

    def printRow(self):
        row = "{}\t{}\t{}\t{}".format(self.eno, self.ename, self.dno, self.phone)
        print(row)

    def saveRow(self):
        with open(file_path+"employee.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.eno, self.ename, self.dno, self.phone])
            print('사원정보 추가 성공')


# ------------------------------------------------------
emp = Employee([11, 'Scott', 20, '010-5784-3210'])
# <__main__.Employee at 0x7cf85b026c50>
# emp.eno, emp.ename, emp.dno, emp.phone
emp.printRow()


# ------------------------------------------------------
def inputEmp():
    eno = input('사번:')
    ename = input('이름:')
    dno = input('부서번호:')
    phone = input('전화번호:')
    return Employee([eno, ename, dno, phone])


# ------------------------------------------------------
def loadEmps():
    emps = []
    with open(file_path+"employee.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            emps.append(Employee(row))
    return emps


# ------------------------------------------------------
# 파일을 로드하고 메모리에서 특정 사원의 정보를 수정하는 예
# 12번 사원의 전화번호를 "010-1111-2222"으로 갱신한다
emps = loadEmps()
for emp in emps:
    if emp.eno == '12':
        emp.phone = '010-1111-2222'
        # 메모리에서 수정된 리스트의 원소를 기존 파일에 덮어쓴다
        # if overwrite(emps):
        #    print('사원정보 수정 성공')
        break
    # emp.printRow()


def showEmps(emps):
    for emp in emps:
        emp.printRow()


# ------------------------------------------------------
def overwrite(emps):
    with open(file_path+"employee.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for emp in emps:
            writer.writerow([emp.eno, emp.ename, emp.dno, emp.phone])
    return True


# ------------------------------------------------------
def updateEmp():
    emps = loadEmps()
    eno = input('수정할 사번:')
    updated = False
    for emp in emps:
        if emp.eno == eno:
            emp.phone = input('새 전화번호:')
            if overwrite(emps):
                updated = True
                print('사원정보 수정 성공')
            break
    if not updated:
        print('사원정보 수정 실패')


# ------------------------------------------------------
# 리스트의 원소 삭제
emps = loadEmps()
for emp in emps:
    if emp.eno == '12':
        emps.remove(emp)
        break

del emps[3]


# ------------------------------------------------------

def deleteEmp():
    emps = loadEmps()
    eno = input('삭제할 사번:')
    deleted = False
    for emp in emps:
        if emp.eno == eno:
            emps.remove(emp)
            if overwrite(emps):
                deleted = True
                print('사원정보 삭제 성공')
            break
    if not deleted:
        print('사원정보 삭제 실패')

# ------------------------------------------------------


def findEmp():
    emps = loadEmps()
    eno = input('검색할 사번:')
    found = False
    for emp in emps:
        if emp.eno == eno:
            emp.printRow()
            found = True
            break
    if not found:
        print('사원정보 검색 실패')


# ------------------------------------------------------
while True:
    menu = input("추가(a), 목록(s), 수정(u), 삭제(d), 검색(f), 종료(x) :")
    if menu == 'a':
        inputEmp().saveRow()
    elif menu == 's':
        showEmps(loadEmps())
    elif menu == 'u':
        updateEmp()
    elif menu == 'd':
        deleteEmp()
    elif menu == 'f':
        findEmp()
    elif menu == 'x':
        print('프로그램 종료...')
        break
    else:
        print("잘못된 메뉴 선택")

print('프로그램 종료됨')
