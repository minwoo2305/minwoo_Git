# 평균을 구하는 멤버함수
# 표준편자를 구하는 멤버함수
# 전체인원의 성적을 정렬해서 등수를 구하는 멤버함수, 이 함수는 2차원 리스트를 반환
# 이차원리스트는 nx3의 2차원 리스트 첫번째 컬럼은 이름, 컬럼성적, 컬럼등수,sort사용x

import math


class Student:
    zero = 0
    z = 0

    def __init__(self, name, grade, count, x, lst):
        self.name = name
        self.grade = grade
        self.count = count
        self.lst = lst
        self.x = x

    def add(self):
        global lst
        self.lst[self.x][0] = self.name
        self.lst[self.x][1] = self.grade
        return lst

    def average(self):
        Student.zero += int(self.lst[self.x][1])
        global average
        average = Student.zero / self.count
        return average

    def seperate(self):
        Student.z += (int(self.lst[self.x][1]) - average) ** 2
        return Student.z


count = int(input("학생수를 입력하시오:"))
lst = [[0] * 3 for i in range(count)]
avg = 0
sep = 0

for x in range(count):
    name = input("이름을 입력하시오 : ")
    grade = input("성적을 입력하시오 : ")
    std = Student(name, grade, count, x, lst)
    std.add()
    avg = std.average()
    sep = std.seperate()


def bubble_sort(lst, count):
    for i in range(count):
        for j in range(count - 1):
            if lst[j][1] < lst[j + 1][1]:
                lst[j][1], lst[j + 1][1] = lst[j + 1][1], lst[j][1]
                lst[j][0], lst[j + 1][0] = lst[j + 1][0], lst[j][0]
    for j in range(len(lst)):
        lst[j][2] = j + 1
    return lst


print("평균:%d" % avg)
print("표준편차:%d" % math.sqrt(sep / count))
bubble_sort(lst, count)
for i in range(count):
    print("이름 : %s | 점수 : %s | 순위 : %s" % (lst[i][0], lst[i][1], lst[i][2]))