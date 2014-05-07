
class A(object):
    def __init__(self):
        """
        Init class A.
        """
        print 'Class A.'

    def fun(self):
        print "A's fun."

class B(object):
    def __init__(self):
        """
        Init class B.
        """
        print 'Class B.'

    def _fun(self):
        print "B's fun."
        self.fun()

class C(A, B):
    def __init__(self):
        """
        Init class C.
        """
        A.__init__(self)
        B.__init__(self)
        print 'Class C'

    def fun(self):
        print "C's fun."

if __name__ == "__main__":
    c = C()
    c._fun()
