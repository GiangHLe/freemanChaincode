def similarity(a,b):
    a=str(a)
    b=str(b)
    La=len(a)
    Lb=len(b)
    
# check if length a not equal b , then add one more character into the string. I chose 'a'
    if La>Lb:
        b=b+(La-Lb)*'a'
        b=str(b)
        Lb=len(b)

    else:
        a=a+(Lb-La)*'a'
        a=str(a)
        La=len(a) 
#check again if the two lengths are not equal then raise the error
    if La!=Lb:
        return -1

    eq=[]   # eq list contain the same characters between 2 lists 
    for i,j in zip(a,b):
        if i==j:
            eq.append(i)
        else:
            pass
    alpha = len(eq)
    beta = max(La,Lb)-alpha  # beta is the difference between two list
    R=0 
    if beta ==0:
        R= "inf"
    else:
        R= alpha/beta
    return R
        