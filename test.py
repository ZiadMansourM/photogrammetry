class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

def change_age(person):
    person.age = 30

p = Person("John", 25)
print(p.age) # Output: 25

change_age(p)
print(p.age) # Output: 30
