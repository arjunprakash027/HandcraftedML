import LinearModels as lm

logreg = lm.LogisticRegression()

print(logreg.fit([[2,3,4,6],[2,3,4,5]],[0,1,0,1],0.1,100))
print(logreg.predict([1,2,3]))