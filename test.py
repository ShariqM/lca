import matplotlib.pyplot as plt

classes = ['A','A','B','C','C','C']
colours = ['r','r','b','g','g','g']
for (i,cla) in enumerate(set(classes)):
    xc = [p for (j,p) in enumerate(x) if classes[j]==cla]
    yc = [p for (j,p) in enumerate(y) if classes[j]==cla]
    cols = [c for (j,c) in enumerate(colours) if classes[j]==cla]
    plt.scatter(xc,yc,c=cols,label=cla)
plt.legend(loc=4)
