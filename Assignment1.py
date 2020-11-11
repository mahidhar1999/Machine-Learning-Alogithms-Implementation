exp=[10,3]
sal=[90,12]
coeff=0.8
xy=coeff*sal[1]/exp[1]
yx=coeff*exp[1]/sal[1]
a=str(xy)
b=str(sal[0]-xy*exp[0])
print("line1 is  "+"y="+a+"x +"+b)
a1=str(yx)
b1=str(exp[0]-yx*sal[0])
print("line2 is  "+"x="+a1+"y "+b1)

print("1.To find sales  2.To find budget")
n=int(input())
if(n==1):
    print("enter budget")
    bud=int(input())
    print(xy*bud+sal[0]-xy*exp[0])
if(n==2):
    print("enter sales target")
    sale=int(input())
    print(yx*sale+exp[0]-yx*sal[0])