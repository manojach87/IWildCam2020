# List definitions
l=[1,2,3,4]
l1=list([1,2,3,4])
l2=list()
l2.append(1)
l2.append(2)
l2.append(3)
l2.append(4)
l3=list((1,2,3,4))

# Set definitions
s={1,2,3,4}
s1=set([1,2,3,4])
s2 = set()
s2.add(1)
s2.add(2)
s2.add(3)
s2.add(4)
s3=set((1,2,3,4))

# Tuple definitions
t=(1,2,3,4)
t1=tuple([1,2,3,4])
t2=tuple((1,2,3,4))
t3=tuple({1,2,3,4})

# Dictionary Definitions
d={"fname":"sgdfhjsdg","lname":"Acharya"}
d.update({"kk":"HHH"})
d["fname"]="Manoj"

print("Lists")
print(l)
print(l1)
print(l2)
print(l3)
print("Sets")
print(s)
print(s1)
print(s2)
print(s3)
print("Tuples")
print(t)
print(t1)
print(t2)
print(t3)
print("Dictionary")
print(d)
print(d.keys())

print(type(d))

a = [1,2,3,4,5]
b = [5,4,3,2,1]

print(zip(a,b))

a = [1,2,3,4,5]
b = [5,4,3,2,1]

print(list(zip(a,b)))