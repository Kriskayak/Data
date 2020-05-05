'''
Jennifer
CLT hw. with chisquare distribution
'''

size = 10000
plt.figure(1,figsize=(8,6))
dist1 = np.random.chisquare(1,100000)
dist2 = np.random.chisquare(1,100000)
dist3 = np.random.chisquare(1,100000)
dist = (dist1+dist2+dist3)/3.0
hist = plt.hist(dist1,bins=200,density=True,edgecolor='none',alpha=0.35,color='red',label='$d_1$')
hist = plt.hist(dist2,bins=200,density=True,edgecolor='none',alpha=0.35,color='blue',label='$d_2$')
hist = plt.hist(dist3,bins=200,density=True,edgecolor='none',alpha=0.35,color='yellow',label='$d_3$')
hist = plt.hist(dist,bins=200,density=True,edgecolor='none',alpha=0.35,color='green',label='$d_1+d_2+d_3$')
plt.xlim(0,10)
leg = plt.legend()


# Hmmm. It looks like the peak of the resultant distribution has moved to the right some, but it sure doesn't look Gaussian!
# 
# Let's average many, many lognormal distributions and see what happens...

size = 10000
ndist = 100000
dist =  np.zeros(size)
pbar = tqdm(desc='Averaging distributions',unit='dists',total=ndist)
for i in range(ndist):
    dist += np.random.chisquare(1,size)
    pbar.update(1)
pbar.close()

dist /= np.float(ndist)

hist = plt.hist(dist,bins=100,density=True,edgecolor='none')


# Holy moly! It sure looks Gaussian. But is it really?

hist = plt.hist(dist,bins=100,density=True,edgecolor='none')
x,y = gauss(x0=np.mean(dist),sig=np.std(dist))
plt.plot(x,y,'r--')
xlim = plt.xlim(0.98,1.02)


'''
js comments
-----------
 - You should really clean up this code more. Text editing is a good skill that needs practice

 - Some of your commands are only relevant for jupyter notebooks

 - 8/10


'''
