__author__ = 'zhouguangfu'

class Parent:
    def __init__(self):
        print 'Parent constructor.'

    def fun(self):
        return self.func()

class Child(Parent):
    def __init__(self):
        Parent.__init__(self)
        print 'Child constructor.'

    def func(self):
        print 'Child func called.'

if __name__ == '__main__':
    child = Child()
    child.fun()